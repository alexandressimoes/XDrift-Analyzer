# Configurar PYTHONPATH para importar a biblioteca XAdapt-Drift
import sys
import os
from pathlib import Path
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

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




# 🔧 CLASSE 1: DriftMetricsCalculator - Envelope para Métricas de Drift
# ===================================================================


import warnings
warnings.filterwarnings('ignore')

class DriftMetricsCalculator:
    """
    Classe especializada para cálculo de métricas de drift.
    Responsabilidades:
    - Calcular todas as métricas estatísticas de drift
    - Fornecer métodos utilitários (bins, normalização)
    - Manter consistência entre diferentes métricas
    - Integrar com SmartDriftAnalyzer para calcular apenas métricas aplicáveis
    """

    # Constante global para evitar divisão por zero
    EPSILON = 1e-10  # IEEE 754 double precision: ~2.22e-16

    def __init__(self, feature_names=None, default_bins='auto'):
        self.feature_names = feature_names
        self.default_bins = default_bins
        
        # Mapeamento de nomes de métricas para métodos
        self.metric_methods = {
            'psi': self.psi,  
            'ks_test': self.kolmogorov_smirnov,
            'chi_square': self.chi_square, 
            'wasserstein_distance': self.wasserstein_distance_metric,
            'hellinger_distance': self.hellinger_distance,
            'tvd': self.total_variation_distance,
            'js_divergence': self.jensen_shannon_divergence,
            'kl_divergence': self.kl_divergence
        }

    def _apply_epsilon(self, probabilities: np.ndarray) -> np.ndarray:
        """Método centralizado para aplicar epsilon"""
        return np.where(probabilities == 0, self.EPSILON, probabilities)
    
    def calculate_doane_bins(self, data):
        """
        Implementa a regra de Doane para cálculo ótimo de bins
        Mais robusta que a regra de Sturges para distribuições assimétricas
        
        Args:
            data: Array ou Series com os dados
            
        Returns:
            int: Número ótimo de bins calculado pela regra de Doane
        """
        # Converter para numpy array e remover NaN
        data_clean = np.array(data)
        data_clean = data_clean[~np.isnan(data_clean)]
        
        n = len(data_clean)
        if n < 3:
            return 3
        
        # Calcular skewness
        skewness = stats.skew(data_clean)
        
        # Calcular número de bins usando regra de Doane
        sigma_g1 = np.sqrt((6 * (n - 2)) / ((n + 1) * (n + 3)))
        
        # Fórmula de Doane: bins = 1 + log2(n) + log2(1 + |g1|/σ_g1)
        bins = 1 + np.log2(n) + np.log2(1 + abs(skewness) / sigma_g1)
        
        # Garantir que seja um número inteiro positivo e razoável
        bins = max(3, min(50, int(np.ceil(bins))))  # Limitado entre 3 e 50
        
        return bins
    
    def _prepare_categorical_data(self, reference, current):
        """
        Converte dados categóricos (strings) em uma representação numérica consistente.
        Retorna as contagens de frequência para cada categoria.
        """
        ref_series = pd.Series(reference).dropna()
        curr_series = pd.Series(current).dropna()

        # Encontrar todas as categorias únicas em ambos os datasets
        all_categories = pd.Index(ref_series.unique()).union(curr_series.unique())

        # Calcular contagens de frequência, garantindo que todas as categorias estejam presentes
        ref_counts = ref_series.value_counts().reindex(all_categories, fill_value=0)
        curr_counts = curr_series.value_counts().reindex(all_categories, fill_value=0)

        return ref_counts, curr_counts, all_categories
    

    def _calculate_histogram_pair(self, reference, current, bins=None, method='doane'):
        """
        Método centralizado para calcular histogramas de forma consistente
        
        Args:
            reference: Dados de referência
            current: Dados atuais
            bins: Número de bins ou None para usar método automático
            method: 'doane', 'sturges', 'scott', 'fd' ou número específico
            
        Returns:
            dict: Contém histogramas, edges, e metadata
        """
        # Converter para numpy arrays e limpar
        ref_vals = np.array(reference)
        curr_vals = np.array(current)
        
        ref_vals = ref_vals[~np.isnan(ref_vals)]
        curr_vals = curr_vals[~np.isnan(curr_vals)]
        
        if len(ref_vals) == 0 or len(curr_vals) == 0:
            raise ValueError("Dados vazios após limpeza de NaN")
        
        # Determinar número de bins
        if bins is None:
            if method == 'doane':
                # Usar Doane nos dados de referência (mais conservador)
                bins = self.calculate_doane_bins(ref_vals)
            elif method == 'sturges':
                bins = int(np.ceil(np.log2(len(ref_vals)) + 1))
            elif method == 'scott':
                h = 3.5 * np.std(ref_vals) / (len(ref_vals) ** (1/3))
                bins = int(np.ceil((ref_vals.max() - ref_vals.min()) / h))
            elif method == 'fd':  # Freedman-Diaconis
                q75, q25 = np.percentile(ref_vals, [75, 25])
                h = 2 * (q75 - q25) / (len(ref_vals) ** (1/3))
                bins = int(np.ceil((ref_vals.max() - ref_vals.min()) / h))
            else:
                bins = 10  # Fallback
        
        # Garantir bins válidos
        bins = max(3, min(50, int(bins)))
        
        # Definir edges baseados na distribuição conjunta para consistência
        all_vals = np.concatenate([ref_vals, curr_vals])
        min_val = all_vals.min()
        max_val = all_vals.max()
        
        # Evitar bins com largura zero
        if max_val == min_val:
            max_val = min_val + 1e-10
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Calcular histogramas
        ref_hist, _ = np.histogram(ref_vals, bins=bin_edges)
        curr_hist, _ = np.histogram(curr_vals, bins=bin_edges)
        
        # Calcular densidades normalizadas
        ref_density, _ = np.histogram(ref_vals, bins=bin_edges, density=True)
        curr_density, _ = np.histogram(curr_vals, bins=bin_edges, density=True)
        
        # Calcular probabilidades (soma = 1)
        ref_prob = ref_hist / np.sum(ref_hist)
        curr_prob = curr_hist / np.sum(curr_hist)
        
        return {
            'ref_hist': ref_hist,
            'curr_hist': curr_hist,
            'ref_density': ref_density,
            'curr_density': curr_density,
            'ref_prob': ref_prob,
            'curr_prob': curr_prob,
            'bin_edges': bin_edges,
            'bins_used': bins,
            'method': method,
            'bin_width': (max_val - min_val) / bins
        }
    
    def get_binning_summary(self, data_list, methods=['doane', 'sturges', 'scott', 'fd']):
        """
        Compara diferentes métodos de binning para um conjunto de dados
        
        Args:
            data_list: Lista de arrays de dados
            methods: Lista de métodos para comparar
            
        Returns:
            dict: Comparação dos métodos de binning
        """
        summary = {}
        
        for method in methods:
            bins_for_method = []
            for data in data_list:
                if method == 'doane':
                    bins = self.calculate_doane_bins(data)
                elif method == 'sturges':
                    bins = int(np.ceil(np.log2(len(data)) + 1))
                elif method == 'scott':
                    h = 3.5 * np.std(data) / (len(data) ** (1/3))
                    bins = int(np.ceil((np.max(data) - np.min(data)) / h))
                elif method == 'fd':
                    q75, q25 = np.percentile(data, [75, 25])
                    h = 2 * (q75 - q25) / (len(data) ** (1/3))
                    bins = int(np.ceil((np.max(data) - np.min(data)) / h))
                else:
                    bins = 10
                
                bins_for_method.append(max(3, min(50, bins)))
            
            summary[method] = {
                'bins': bins_for_method,
                'mean_bins': np.mean(bins_for_method),
                'std_bins': np.std(bins_for_method)
            }
        
        return summary
    
    def chi_square(self, reference, current, column_type, bins='auto'):
        """
        Calcula o drift usando o teste Qui-Quadrado.
        - Para Categóricos: Usa o Teste de Homogeneidade em uma tabela de contingência 2xN.
        - Para Numéricos: Usa o Teste de Aderência (Goodness-of-Fit) em dados binarizados.
        """
        ref_series = pd.Series(reference).dropna()
        curr_series = pd.Series(current).dropna()

        if ref_series.empty or curr_series.empty:
            return {'error': 'Dados de referência ou atuais estão vazios após remover NaNs.'}

        try:
            if column_type == 'categorical':
                ref_counts, curr_counts, categories = self._prepare_categorical_data(ref_series, curr_series)
                contingency_table = pd.DataFrame([ref_counts, curr_counts])
                chi2_stat, p_value, df, _ = stats.chi2_contingency(contingency_table)
                
                return {
                    'chi2_statistic': float(chi2_stat),
                    'p_value': float(p_value),
                    'degrees_of_freedom': df,
                    'method': 'chi2_homogeneity_test'
                }
            
            elif column_type == 'numerical':
                num_bins = self.calculate_doane_bins(ref_series) if bins == 'auto' else bins
                hist_data = self._calculate_histogram_pair(ref_series, curr_series, bins=num_bins)
                contingency_table = np.array([hist_data['ref_hist'], hist_data['curr_hist']])
                chi2_stat, p_value, df, _ = stats.chi2_contingency(contingency_table)

                return {
                    'chi2_statistic': float(chi2_stat),
                    'p_value': float(p_value),
                    'degrees_of_freedom': df,
                    'method': 'chi2_on_binned_data',
                    'bins_used': hist_data['bins_used']
                }
            else:
                return {'error': f'Tipo de coluna desconhecido: {column_type}'}

        except Exception as e:
            return {'error': f'Falha no cálculo do Qui-Quadrado: {e}'}

    def wasserstein_distance_metric(self, reference, current, column_type):
        """
        Implementa distância de Wasserstein (Earth Mover's Distance)
        Mede o "custo" mínimo para transformar uma distribuição na outra
        """
        try:
            ref_vals = np.array(reference)
            curr_vals = np.array(current)
            
            # Remover NaN
            ref_vals = ref_vals[~np.isnan(ref_vals)]
            curr_vals = curr_vals[~np.isnan(curr_vals)]
            
            # Calcular distância de Wasserstein
            wasserstein_dist = wasserstein_distance(ref_vals, curr_vals)
            
            # Calcular escala robusta (IQR) para normalização
            ref_q75 = np.percentile(ref_vals, 75)
            ref_q25 = np.percentile(ref_vals, 25)
            reference_iqr = ref_q75 - ref_q25
            
            # Normalizar pela amplitude dos dados para interpretação
            data_range = max(ref_vals.max(), curr_vals.max()) - min(ref_vals.min(), curr_vals.min())
            if data_range == 0:
                data_range = 1e-10
                
            normalized_distance = wasserstein_dist / data_range
            
            # Normalização por IQR (mais robusta a outliers)
            if reference_iqr > 1e-9:
                normalized_by_iqr = wasserstein_dist / reference_iqr
            else:
                normalized_by_iqr = normalized_distance
            
            return {
                'wasserstein_distance': float(wasserstein_dist),
                'normalized_distance': float(normalized_distance),
                'normalized_by_iqr': float(normalized_by_iqr),
                'data_range': float(data_range),
                'reference_iqr': float(reference_iqr)
            }
            
        except Exception as e:
            return {
                'wasserstein_distance': np.nan,
                'normalized_distance': np.nan,
                'error': str(e)
            }
    
    def hellinger_distance(self, reference, current, column_type, bins=None):
        """
        Implementa distância de Hellinger. Funciona para numéricos e categóricos.
        """
        try:
            if column_type == 'categorical':
                ref_counts, curr_counts, _ = self._prepare_categorical_data(reference, current)
                ref_total = np.sum(ref_counts)
                curr_total = np.sum(curr_counts)
                ref_prob = ref_counts / ref_total if ref_total > 0 else np.zeros_like(ref_counts, dtype=float)
                curr_prob = curr_counts / curr_total if curr_total > 0 else np.zeros_like(curr_counts, dtype=float)
                bins_used = len(ref_counts)
            else: # numerical
                hist_data = self._calculate_histogram_pair(reference, current, bins, 'doane')
                ref_prob = hist_data['ref_prob']
                curr_prob = hist_data['curr_prob']
                bins_used = hist_data['bins_used']

            ref_prob = self._apply_epsilon(ref_prob)
            curr_prob = self._apply_epsilon(curr_prob)

            hellinger_dist = np.sqrt(0.5 * np.sum((np.sqrt(ref_prob) - np.sqrt(curr_prob)) ** 2))
            
            return {
                'hellinger_distance': float(hellinger_dist),
                'bins_used': bins_used,
                'method': 'categorical_counts' if column_type == 'categorical' else 'doane_binning'
            }
        except Exception as e:
            return {'hellinger_distance': np.nan, 'error': str(e)}
    
    def total_variation_distance(self, reference, current, column_type, bins=None):
        """Implementa Total Variation Distance (TVD) para ambos os tipos de dados."""
        try:
            if column_type == 'categorical':
                ref_counts, curr_counts, _ = self._prepare_categorical_data(reference, current)
                ref_total = np.sum(ref_counts)
                curr_total = np.sum(curr_counts)
                ref_prob = ref_counts / ref_total if ref_total > 0 else np.zeros_like(ref_counts, dtype=float)
                curr_prob = curr_counts / curr_total if curr_total > 0 else np.zeros_like(curr_counts, dtype=float)
                bins_used = len(ref_counts)
                method = 'categorical_counts'
            else: # numerical
                hist_data = self._calculate_histogram_pair(reference, current, bins, 'doane')
                ref_prob = hist_data['ref_prob']
                curr_prob = hist_data['curr_prob']
                bins_used = hist_data['bins_used']
                method = 'doane_binning'

            # A fórmula do TVD é a mesma para ambos os tipos de probabilidade
            tvd = 0.5 * np.sum(np.abs(ref_prob - curr_prob))
            
            return {
                'tvd': float(tvd),
                'bins_used': bins_used,
                'method': method
            }
        except Exception as e:
            return {'tvd': np.nan, 'error': str(e)}

    def psi(self, 
        reference, 
        current, 
        column_type: str, 
        bins: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        PSI usando método unificado de binning (_calculate_histogram_pair)
        
        Vantagens:
        - Consistência com Hellinger, TVD, JS
        - Bins calculados por Doane nos mesmos dados
        - Edges alinhados entre ref e current
        """
        try:
            if column_type == 'categorical':
                ref_counts, curr_counts, _ = self._prepare_categorical_data(reference, current)
                ref_total = np.sum(ref_counts)
                curr_total = np.sum(curr_counts)
                ref_prop = ref_counts / ref_total if ref_total > 0 else np.zeros_like(ref_counts, dtype=float)
                curr_prop = curr_counts / curr_total if curr_total > 0 else np.zeros_like(curr_counts, dtype=float)
                bins_used = len(ref_counts)
                method = 'categorical_counts'
            
            else:  # numerical - USAR _calculate_histogram_pair
                if bins is None or bins == 'auto':
                    bins = self.calculate_doane_bins(reference)
                
                # UNIFICADO: usar método centralizado
                hist_data = self._calculate_histogram_pair(
                    reference, current, bins, method='doane'
                )
                
                ref_prop = hist_data['ref_prob']
                curr_prop = hist_data['curr_prob']
                bins_used = hist_data['bins_used']
                method = 'doane_unified'
            
            ref_prop = self._apply_epsilon(ref_prop)
            curr_prop = self._apply_epsilon(curr_prop)
            
            # Cálculo PSI padrão
            psi_value = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))

            return {
                'psi_value': float(psi_value),
                'regulatory_compliant': True,
                'bins_used': bins_used,
                'method': method,
                'binning_consistency': 'unified_with_other_metrics'  # NOVO
            }
        except Exception as e:
            return {'psi_value': np.nan, 'error': str(e)}
    
    def kolmogorov_smirnov(self, reference, current, column_type):
        """
        Implementa KS Test com interpretação estatística formal
        Não usa binning - trabalha diretamente com CDFs
        """
        try:
            ref_vals = np.array(reference)
            curr_vals = np.array(current)
            
            # Remover NaN
            ref_vals = ref_vals[~np.isnan(ref_vals)]
            curr_vals = curr_vals[~np.isnan(curr_vals)]
            
            ks_stat, ks_p = stats.ks_2samp(ref_vals, curr_vals)
            
            return {
                'ks_statistic': float(ks_stat),
                'p_value': float(ks_p),
                'method': 'cdf_based'
            }
            
        except Exception as e:
            return {
                'ks_statistic': np.nan,
                'p_value': np.nan,
                'error': str(e)
            }

    def jensen_shannon_divergence(self, reference, current, column_type, bins=None):
        """
        Calcula a divergência Jensen-Shannon. Funciona para numéricos e categóricos.
        """
        try:
            if column_type == 'categorical':
                ref_counts, curr_counts, _ = self._prepare_categorical_data(reference, current)
                ref_total = np.sum(ref_counts)
                curr_total = np.sum(curr_counts)
                ref_prob = ref_counts / ref_total if ref_total > 0 else np.zeros_like(ref_counts, dtype=float)
                curr_prob = curr_counts / curr_total if curr_total > 0 else np.zeros_like(curr_counts, dtype=float)
                bins_used = len(ref_counts)
            else: # numerical
                hist_data = self._calculate_histogram_pair(reference, current, bins, 'doane')
                ref_prob = hist_data['ref_prob']
                curr_prob = hist_data['curr_prob']
                bins_used = hist_data['bins_used']

            ref_prob = self._apply_epsilon(ref_prob)
            curr_prob = self._apply_epsilon(curr_prob)
            
            js_div = jensenshannon(ref_prob, curr_prob) ** 2
            
            return {
                'js_divergence': float(js_div),
                'bins_used': bins_used,
                'method': 'categorical_counts' if column_type == 'categorical' else 'doane_binning'
            }
        except Exception as e:
            return {'js_divergence': np.nan, 'error': str(e)}
    

    def kl_divergence(self, reference, current, column_type, bins=None):
        """
        Calcula KL Divergence. Funciona para numéricos e categóricos.
        """
        try:
            if column_type == 'categorical':
                ref_counts, curr_counts, _ = self._prepare_categorical_data(reference, current)
                ref_total = np.sum(ref_counts)
                curr_total = np.sum(curr_counts)
                ref_prob = ref_counts / ref_total if ref_total > 0 else np.zeros_like(ref_counts, dtype=float)
                curr_prob = curr_counts / curr_total if curr_total > 0 else np.zeros_like(curr_counts, dtype=float)
                bins_used = len(ref_counts)
            else: # numerical
                hist_data = self._calculate_histogram_pair(reference, current, bins, 'doane')
                ref_prob = hist_data['ref_prob']
                curr_prob = hist_data['curr_prob']
                bins_used = hist_data['bins_used']

            ref_prob = self._apply_epsilon(ref_prob)
            curr_prob = self._apply_epsilon(curr_prob)
            
            kl_div = np.sum(curr_prob * np.log(curr_prob / ref_prob))
            
            return {
                'kl_divergence': float(kl_div),
                'bins_used': bins_used,
                'method': 'categorical_counts' if column_type == 'categorical' else 'doane_binning'
            }
        except Exception as e:
            return {'kl_divergence': np.nan, 'error': str(e)}

    def calculate_metrics_for_feature(self, 
                                           reference_data, 
                                           current_data, 
                                           applicable_metrics,
                                           column_type,
                                           feature_name=None):
        """
        Calcula apenas as métricas aplicáveis indicadas pelo DriftAnalyzer
        
        Args:
            reference_data: Dados de referência (array-like)
            current_data: Dados atuais (array-like)
            applicable_metrics: Lista de métricas aplicáveis do DriftAnalyzer
            feature_name: Nome da feature (opcional)
            
        Returns:
            dict: Dicionário com apenas as métricas aplicáveis calculadas
        """
        metrics_results = {}
        calculated_count = 0
        skipped_count = 0
        
        try:
            for metric_name in applicable_metrics:
                # Verificar se temos método para esta métrica
                if metric_name in self.metric_methods:
                    method = self.metric_methods[metric_name]
                    try:
                        result = method(reference_data, current_data, column_type)
                        metrics_results[metric_name] = result
                        calculated_count += 1
                    except Exception as e:
                        metrics_results[metric_name] = {'error': str(e)}
                        skipped_count += 1
                else:
                    metrics_results[metric_name] = {'error': f'Método não implementado: {metric_name}'}
                    skipped_count += 1
            
            # Adicionar metadata sobre o cálculo
            metrics_results['_metadata'] = {
                'feature_name': feature_name,
                'total_applicable': len(applicable_metrics),
                'calculated': calculated_count,
                'skipped': skipped_count,
                'applicable_metrics': applicable_metrics,
                'calculation_mode': 'smart_selective'
            }
            
        except Exception as e:
            metrics_results['calculation_error'] = str(e)
        
        return metrics_results

    def calculate_metrics_from_report(self, 
                                           reference_df, 
                                           current_df, 
                                           analysis_report):
        """
        Calcula métricas para múltiplas features usando relatório do DriftAnalyzer
        
        Args:
            reference_df: DataFrame de referência
            current_df: DataFrame atual
            analysis_report: Relatório de saída do DriftAnalyzer
            
        Returns:
            dict: Resultados organizados por feature
        """
        results = {}
        
        if 'columns' not in analysis_report:
            return {'error': 'Formato de relatório inválido - chave "columns" não encontrada'}
        
        columns_info = analysis_report['columns']
        
        
        for feature_name, feature_info in columns_info.items():
            # Verificar se feature existe nos dataframes
            if feature_name not in reference_df.columns or feature_name not in current_df.columns:
                results[feature_name] = {
                    'error': f'Feature {feature_name} não encontrada nos DataFrames'
                }
                continue
            
            # Obter métricas aplicáveis
            applicable_metrics = feature_info.get('applicable_metrics', [])
            column_type = feature_info.get('feature_type', 'unknown')
            
            # print(f"{feature_name} ({column_type}): {len(applicable_metrics)} métricas")
            
            # Calcular métricas aplicáveis
            feature_results = self.calculate_metrics_for_feature(
                reference_data=reference_df[feature_name],
                current_data=current_df[feature_name],
                applicable_metrics=applicable_metrics,
                column_type=column_type,
                feature_name=feature_name
            )
            
            # Adicionar informações do tipo de coluna
            feature_results['column_type'] = column_type
            feature_results['metric_details'] = feature_info.get('metric_details', {})
            
            results[feature_name] = feature_results
        
        # Resumo geral
        total_features = len(results)
        successful_features = len([f for f in results.values() if 'error' not in f])
        total_metrics_calculated = sum(
            r.get('_metadata', {}).get('calculated', 0) 
            for r in results.values() 
            if '_metadata' in r
        )
        
        
        # Adicionar resumo aos resultados
        results['_summary'] = {
            'total_features': total_features,
            'successful_features': successful_features,
            'total_metrics_calculated': total_metrics_calculated,
            'source_report': analysis_report.get('summary', {}),
            'calculation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        return results

    def calculate_all_metrics_for_feature(self, 
                                          reference_data, 
                                          current_data, 
                                          feature_name=None):
        """
        Calcula todas as métricas disponíveis para uma feature específica
        (Método legado mantido para compatibilidade)
        
        Args:
            reference_data: Dados de referência (array-like)
            current_data: Dados atuais (array-like)
            feature_name: Nome da feature (opcional)
            
        Returns:
            dict: Dicionário com todas as métricas calculadas
        """
        # Lista de todas as métricas disponíveis
        all_metrics = list(self.metric_methods.keys())
        
        return self.calculate_metrics_for_feature(
            reference_data=reference_data,
            current_data=current_data,
            applicable_metrics=all_metrics,
            feature_name=feature_name
        )
    