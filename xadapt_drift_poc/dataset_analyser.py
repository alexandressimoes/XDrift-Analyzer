# Configurar PYTHONPATH para importar a biblioteca XAdapt-Drift
import sys
import os
from pathlib import Path
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Union

import shap

# Importando o método de Permutation Importance
from sklearn.inspection import permutation_importance

# Imports para criar um modelo de exemplo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy import stats
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
import sys
plt.style.use('seaborn-v0_8-pastel')
sns.set_palette('pastel')

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')







# CLASSE: DatasetAnalyzer - Classe Auxiliar para Análise de Estatisticas Basicas de métricas aplicáveis a cada feature


class DatasetAnalyzer:
    """
    Classe responsável por analisar datasets e recomendar métodos de detecção de drift.
    
    Funcionalidades principais:
    - Análise estatística detalhada de features (sempre retornada)
    - Recomendações de métricas de drift (opcional via flag booleana)
    - Separação clara entre análise e sugestões
    """
    def __init__(self, model=None, target_type='classification'):
        self.model = model
        self.target_type = target_type
        
        # Métricas por tipo de feature
        self.applicable_metrics = {
        'numerical': ['psi', 'ks_test', 'wasserstein_distance', 'hellinger_distance', 'js_divergence', 'kl_divergence'],
        'categorical': ['psi', 'chi_square', 'hellinger_distance', 'js_divergence', 'kl_divergence'],
        'boolean': ['psi', 'chi_square'],
        'datetime': ['psi'] # PSI pode ser usado em features extraídas (ex: dia da semana)
    }

    
    @staticmethod
    def detect_column_types(
        df: pd.DataFrame, 
        cardinality_threshold: int = 10,
        unique_ratio_threshold: float = 0.05,
        user_defined_types: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, str]:
        """
        Detecta o tipo semântico de cada coluna para análise de drift.

        Classifica as colunas em: 'numerical', 'categorical', 'boolean', 'datetime'.
        Uma coluna numérica é classificada como 'categorical' se sua cardinalidade
        ou razão de valores únicos estiver abaixo dos limiares.

        Args:
            df (pd.DataFrame): O DataFrame a ser analisado.
            cardinality_threshold (int): Limite absoluto de valores únicos para
                                         considerar uma coluna numérica como categórica.
            unique_ratio_threshold (float): Limite de razão (únicos / total) para
                                            considerar uma coluna numérica como categórica.
            user_defined_types (Optional[Dict[str, List[str]]]): Dicionário para forçar
                tipos para colunas específicas. Ex: {'categorical': ['user_id']}.

        Returns:
            Dict[str, str]: Um dicionário mapeando nome da coluna ao seu tipo detectado.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("O input deve ser um pandas DataFrame.")

        column_types = {}
        
        # Aplicar tipos definidos pelo usuário primeiro para sobrepor a detecção automática
        if user_defined_types:
            for type_name, columns in user_defined_types.items():
                for col in columns:
                    if col in df.columns:
                        column_types[col] = type_name

        for column in df.columns:
            # Pular colunas que já foram definidas pelo usuário
            if column in column_types:
                continue

            col_data = df[column]
            
            # 1. Checar por tipo booleano
            if pd.api.types.is_bool_dtype(col_data):
                column_types[column] = 'boolean'
                continue

            # 2. Checar por tipo datetime
            if pd.api.types.is_datetime64_any_dtype(col_data):
                column_types[column] = 'datetime'
                continue

            # 3. Checar por tipo numérico (e diferenciar entre numérico e categórico)
            if pd.api.types.is_numeric_dtype(col_data):
                non_null_count = col_data.count()
                if non_null_count == 0:
                    # Se a coluna estiver toda nula, podemos classificá-la como numérica por padrão
                    # ou ignorá-la. Ignorar pode ser mais seguro.
                    continue
                
                unique_count = col_data.nunique()
                unique_ratio = unique_count / non_null_count
                
                if unique_count <= cardinality_threshold or unique_ratio <= unique_ratio_threshold:
                    column_types[column] = 'categorical'
                else:
                    column_types[column] = 'numerical'
            
            # 4. O restante é considerado categórico por padrão
            else:
                column_types[column] = 'categorical'
                
        return column_types
    
    def _estimate_outlier_rate(self, data: pd.Series) -> float:
        """Estima taxa de outliers usando IQR de forma segura."""
        if not pd.api.types.is_numeric_dtype(data):
            return 0.0
        try:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            # Evitar erros com IQR = 0
            if IQR == 0:
                return 0.0
            outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
            return outliers / len(data.dropna())
        except (TypeError, ValueError):
            self.logger.warning(f"Não foi possível calcular taxa de outliers para a série '{data.name}'.")
            return 0.0

    def _get_applicable_metrics(self, column_type, sample_size):
        """Determina quais métricas são aplicáveis para uma coluna específica"""
        applicable_metrics = []
        metric_info = {}
        
        # PSI - aplicável para todos os tipos
        if sample_size >= 50:
            applicable_metrics.append('psi')
            metric_info['psi'] = {'reason': 'Padrão regulatório, funciona com binning'}
        
        # KL/JS Divergence - melhor para dados contínuos
        if sample_size >= 100:
            applicable_metrics.extend(['kl_divergence', 'js_divergence'])
            metric_info['kl_divergence'] = {'reason': 'Sensível a mudanças distribucionais'}
            metric_info['js_divergence'] = {'reason': 'Versão simétrica e mais robusta da KL'}
        
        # KS Test - apenas para dados contínuos
        if column_type == 'numerical' and sample_size >= 30:
            applicable_metrics.append('ks_test')
            metric_info['ks_test'] = {'reason': 'Teste estatístico formal para dados contínuos'}
        
        # Chi-squared - para dados categóricos
        if column_type == "categorical" and sample_size >= 50:
            applicable_metrics.append('chi_square')
            metric_info['chi_square'] = {'reason': 'Teste estatístico para dados categóricos'}
        
        # Hellinger Distance - aplicável para todos os tipos
        if sample_size >= 50:
            applicable_metrics.append('hellinger_distance')
            metric_info['hellinger_distance'] = {'reason': 'Métrica robusta baseada em distância'}
        
        # Wasserstein Distance - melhor para dados contínuos
        if column_type in ['numerical', 'categorical_numeric'] and sample_size >= 50:
            applicable_metrics.append('wasserstein_distance')
            metric_info['wasserstein_distance'] = {'reason': 'Earth Mover Distance para dados ordenados'}
        
        return applicable_metrics, metric_info


    def _generate_drift_suggestions(self, statistical_report):
        """Gera sugestões de métricas de drift baseadas na análise estatística"""
        drift_suggestions = {
            # 'recommended_metrics_by_feature': {},
            'global_monitoring_strategy': {
                'high_priority_features': [],
                'monitoring_frequency': 'weekly',  # default
                'alert_thresholds': {
                    'psi_threshold': 0.2,
                    'chi_square_pvalue': 0.05,
                    'ks_test_pvalue': 0.05,
                    'hellinger_distance': 0.3,
                    'js_divergence': 0.1
                }
            },
            'columns': {}
        }
        
        # Analisar cada feature para gerar sugestões específicas
        for feature, analysis in statistical_report['feature_analysis'].items():
            feature_type = analysis['feature_type']
            sample_size = analysis['basic_statistics']['sample_size']
            
            # Obter métricas aplicáveis
            applicable_metrics, metric_info = self._get_applicable_metrics(feature_type, sample_size)
            
            # Determinar prioridade baseada em características
            priority = 'MEDIUM'  # default
            
            if analysis['comparison_analysis']:
                comparison = analysis['comparison_analysis']
                
                # Alta prioridade se houve mudança de tipo
                if not comparison['type_consistency']:
                    priority = 'CRITICAL'
                
                # Alta prioridade para categóricas com mudanças significativas
                elif feature_type =='categorical':
                    if 'categorical_changes' in comparison:
                        cat_changes = comparison['categorical_changes']
                        if cat_changes['new_categories'] or cat_changes['missing_categories']:
                            priority = 'HIGH'
                
                # Alta prioridade para numéricas com mudanças grandes
                elif feature_type == 'numerical':
                    if 'numerical_changes' in comparison:
                        num_changes = comparison['numerical_changes']
                        if abs(num_changes['mean_change_pct']) > 20 or abs(num_changes['std_change_pct']) > 30:
                            priority = 'HIGH'
            
            # Armazenar sugestões para a feature ['recommended_metrics_by_feature']
            drift_suggestions['columns'][feature] = {
                'feature_type': feature_type,
                'applicable_metrics': applicable_metrics,
                # 'metric_details': metric_info,
                # 'monitoring_priority': priority,
                # 'sample_size': sample_size
            }
            
            # Adicionar às features de alta prioridade se necessário
            if priority in ['HIGH', 'CRITICAL']:
                drift_suggestions['global_monitoring_strategy']['high_priority_features'].append({
                    'feature': feature,
                    'priority': priority,
                    'reason': self._get_priority_reason(analysis, feature_type)
                })
        
        return drift_suggestions
    
    def _get_priority_reason(self, analysis, feature_type):
        """Determina a razão da prioridade de monitoramento"""
        if analysis['comparison_analysis']:
            comparison = analysis['comparison_analysis']
            
            if not comparison['type_consistency']:
                return f"Mudança de tipo detectada: {comparison['detected_types']['reference']} → {comparison['detected_types']['current']}"
            
            if feature_type == 'categorical' and 'categorical_changes' in comparison:
                cat_changes = comparison['categorical_changes']
                if cat_changes['new_categories']:
                    return f"Novas categorias detectadas: {len(cat_changes['new_categories'])} adicionadas"
                if cat_changes['missing_categories']:
                    return f"Categorias perdidas: {len(cat_changes['missing_categories'])} removidas"
            
            if feature_type == 'numerical' and 'numerical_changes' in comparison:
                num_changes = comparison['numerical_changes']
                if abs(num_changes['mean_change_pct']) > 20:
                    return f"Mudança significativa na média: {num_changes['mean_change_pct']:.1f}%"
                if abs(num_changes['std_change_pct']) > 30:
                    return f"Mudança significativa na variabilidade: {num_changes['std_change_pct']:.1f}%"
        
        return "Feature detectada como importante para monitoramento"
    

    def analyze_dataset(self, reference_df, current_df=None, target_column=[], suggest_drift_metrics=False):
        """
        Analisa um dataset e retorna relatório detalhado com tipos de features.
        
        Args:
            reference_df (pd.DataFrame): Dataset de referência
            current_df (pd.DataFrame, optional): Dataset atual para comparação
            target_column (list): Lista de colunas target a serem excluídas da análise
            suggest_drift_metrics (bool): Se True, retorna também sugestões de métricas de drift
        
        Returns:
            tuple: (statistical_report, drift_suggestions) se suggest_drift_metrics=True
                   statistical_report apenas se suggest_drift_metrics=False
        """
        # Relatório de análise estatística
        statistical_report = {
            'dataset_overview': {
                'total_features': len(reference_df.columns),
                'analyzed_features': len([col for col in reference_df.columns if col not in target_column]),
                'excluded_targets': target_column,
                'total_samples': len(reference_df),
                'comparison_available': current_df is not None
            },
            'feature_analysis': {}
        }
        
        # Remover coluna target se especificada
        analysis_columns = [col for col in reference_df.columns if col not in target_column]
        print(f"📊 Analisando {len(analysis_columns)} features (excluindo targets: {target_column})")
        
        # Detectar tipos das features
        reference_feature_types = self.detect_column_types(reference_df[analysis_columns])
        current_feature_types = self.detect_column_types(current_df[analysis_columns]) if current_df is not None else {}
        
        # Contadores por tipo
        type_counts = {'numerical': 0, 'categorical': 0}
        
        for column in analysis_columns:
            # Analisar dados de referência
            ref_data = reference_df[column]
            feature_type = reference_feature_types[column]
            type_counts[feature_type] += 1
            
            print(f"   • {column}: {feature_type}")
            
            # Estatísticas básicas universais
            basic_stats = {
                'data_type': str(ref_data.dtype),
                'unique_values': ref_data.nunique(),
                'null_count': ref_data.isnull().sum(),
                'null_percentage': round((ref_data.isnull().sum() / len(ref_data)) * 100, 2),
                'sample_size': len(ref_data)
            }
            
            # Análise específica por tipo
            type_specific_info = {}
            
            if feature_type == 'categorical':
                categories = ref_data.value_counts().head(10)
                type_specific_info = {
                    'top_categories': categories.to_dict(),
                    'total_categories': ref_data.nunique(),
                    'most_frequent': ref_data.mode().iloc[0] if len(ref_data.mode()) > 0 else None,
                    'category_distribution': {
                        'most_common_pct': round((ref_data.value_counts().iloc[0] / len(ref_data)) * 100, 2) if len(ref_data.value_counts()) > 0 else 0
                    }
                }
            
            elif feature_type == 'numerical':
                type_specific_info = {
                    'central_tendency': {
                        'mean': float(ref_data.mean()),
                        'median': float(ref_data.median())
                    },
                    'dispersion': {
                        'std': float(ref_data.std()),
                        'range': float(ref_data.max() - ref_data.min()),
                        'iqr': float(ref_data.quantile(0.75) - ref_data.quantile(0.25))
                    },
                    'distribution_shape': {
                        'skewness': float(ref_data.skew()),
                        'kurtosis': float(ref_data.kurtosis())
                    },
                    'quartiles': {
                        'q25': float(ref_data.quantile(0.25)),
                        'q50': float(ref_data.quantile(0.50)),
                        'q75': float(ref_data.quantile(0.75))
                    },
                    'extremes': {
                        'min': float(ref_data.min()),
                        'max': float(ref_data.max()),
                        'outlier_rate': self._estimate_outlier_rate(ref_data)
                    }
                }

            # Comparação com dados atuais se disponível
            comparison_analysis = None
            if current_df is not None and column in current_df.columns:
                curr_data = current_df[column]
                curr_type = current_feature_types[column]
                
                comparison_analysis = {
                    'type_consistency': feature_type == curr_type,
                    'detected_types': {'reference': feature_type, 'current': curr_type},
                    'size_comparison': {
                        'reference_size': len(ref_data),
                        'current_size': len(curr_data),
                        'size_change_pct': round(((len(curr_data) - len(ref_data)) / len(ref_data)) * 100, 2)
                    }
                }
                
                # Indicadores básicos de drift por tipo
                if feature_type == 'categorical':
                    # Para categóricos: verificar mudanças nas categorias
                    ref_categories = set(ref_data.unique())
                    curr_categories = set(curr_data.unique())

                    print("DATASET ANALYSIS")
                    print(f"Reference Categories: {ref_categories}")
                    print(f"Current Categories: {curr_categories}")

                    comparison_analysis['categorical_changes'] = {
                        'new_categories': list(curr_categories - ref_categories),
                        'missing_categories': list(ref_categories - curr_categories),
                        'category_count_change': len(curr_categories) - len(ref_categories)
                    }
                
                elif feature_type == 'numerical':
                    # Para numéricos: mudanças estatísticas básicas
                    comparison_analysis['numerical_changes'] = {
                        'mean_change': float(curr_data.mean() - ref_data.mean()),
                        'mean_change_pct': round(((curr_data.mean() - ref_data.mean()) / ref_data.mean()) * 100, 2) if ref_data.mean() != 0 else 0,
                        'std_change': float(curr_data.std() - ref_data.std()),
                        'std_change_pct': round(((curr_data.std() - ref_data.std()) / ref_data.std()) * 100, 2) if ref_data.std() != 0 else 0
                    }
            
            # Armazenar análise da feature
            statistical_report['feature_analysis'][column] = {
                'feature_type': feature_type,
                'basic_statistics': basic_stats,
                'type_specific_analysis': type_specific_info,
                'comparison_analysis': comparison_analysis
            }
        
        # Adicionar resumo da composição do dataset
        statistical_report['dataset_overview']['composition'] = {
            'by_type': type_counts,
            'type_percentages': {
                feature_type: round((count / len(analysis_columns)) * 100, 1) 
                for feature_type, count in type_counts.items()
            }
        }
        
        # Se sugestões de drift não foram solicitadas, retorna apenas análise estatística
        if not suggest_drift_metrics:
            return statistical_report
        
        # Gerar sugestões de métricas de drift
        drift_suggestions = self._generate_drift_suggestions(statistical_report)
        
        return statistical_report, drift_suggestions
    


