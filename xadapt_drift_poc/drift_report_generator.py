# 📊 DriftReportGenerator - Geração de Relatórios Baseada em Evidências Científicas
# =====================================================================================
# Configurar PYTHONPATH para importar a biblioteca XAdapt-Drift
import sys
import os
from pathlib import Path
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from statsmodels.stats.multitest import multipletests

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




class DriftReportGenerator:
    """
    Classe especializada para geração de relatórios de drift baseada em evidências científicas.
    
    Responsabilidades:
    - Integrar estatísticas do ImprovedDatasetDriftAnalyzer
    - Processar métricas do DriftMetricsCalculator
    - Aplicar thresholds baseados em literatura científica
    - Executar análises de impacto no modelo
    - Gerar relatórios com recomendações fundamentadas
    
    Referências Científicas:
    - PSI Thresholds: Narayanan (2010), SAS Institute Guidelines
    - KS Test: Massey Jr. (1951), Kolmogorov (1933)
    - Wasserstein: Vallender (1974), Kantorovich-Rubinstein
    - Hellinger: Hellinger (1909), Beran (1977)
    """
    
    def __init__(self, model=None, feature_names=None):
        self.model = model
        self.feature_names = feature_names if feature_names else []
        
        self.thresholds = {
            'psi': {
                # Baseado em Narayanan (2010) e SAS Institute Guidelines
                'stable': 0.1,        # PSI < 0.1: No significant change
                'moderate': 0.2,      # 0.1 ≤ PSI < 0.2: Moderate change requiring investigation
                'unstable': 0.25      # PSI ≥ 0.25: Major shift requiring immediate action
            },
            'ks_test': {
                # Baseado em análise estatística clássica
                'alpha_strict': 0.01,    # Highly significant
                'alpha_standard': 0.05,  # Significant
                'alpha_liberal': 0.10    # Marginally significant
            },
            'chi_square': {
                # Baseado em Pearson (1900) e Fisher (1922)
                'alpha_strict': 0.01,
                'alpha_standard': 0.05,
                'alpha_liberal': 0.10
            },
            'wasserstein': {
                # Baseado em Earth Mover's Distance literature
                # Normalizado para [0,1] considerando escala dos dados
                'low': 0.1,      # Pequena distância
                'moderate': 0.3, # Distância moderada
                'high': 0.5      # Grande distância
            },
            'hellinger': {
                # Baseado em Hellinger (1909) e Beran (1977)
                # Hellinger distance ∈ [0,1]
                'low': 0.1,      # H < 0.1: Distribuições similares
                'moderate': 0.3, # 0.1 ≤ H < 0.3: Diferença moderada
                'high': 0.5      # H ≥ 0.5: Distribuições muito diferentes
            },
            'js_divergence': {
                # Baseado em Jensen-Shannon divergence literature
                'low': 0.1,
                'moderate': 0.3,
                'high': 0.5
            },
            'kl_divergence': {
                # Baseado em Kullback-Leibler (1951)
                # KL divergence ∈ [0,∞), thresholds context-dependent
                'low': 0.1,
                'moderate': 0.5,
                'high': 1.0
            }
        }
        
        # Contexto para interpretação de métricas por tipo de dados
        self.context_guidelines = {
            'numerical': {
                'primary_metrics': ['ks_test', 'wasserstein', 'hellinger'],
                'secondary_metrics': ['psi', 'js_divergence', 'kl_divergence'],
                'interpretation_notes': 'Métricas de distância são mais adequadas para dados contínuos'
            },
            'categorical': {
                'primary_metrics': ['chi_square', 'psi', 'hellinger'],
                'secondary_metrics': ['js_divergence'],
                'interpretation_notes': 'Chi-squared é gold standard para dados categóricos'
            },
            'categorical_numeric': {
                'primary_metrics': ['chi_square', 'psi', 'hellinger'],
                'secondary_metrics': ['wasserstein', 'js_divergence'],
                'interpretation_notes': 'Tratar como categórico, mas Wasserstein pode capturar ordem'
            }
        }
    
    def integrate_dataset_statistics(self, statistical_report):
        """
        Integra estatísticas básicas do ImprovedDatasetDriftAnalyzer
        
        Args:
            statistical_report: Output do analyze_dataset() do ImprovedDatasetDriftAnalyzer
        
        Returns:
            dict: Estatísticas estruturadas por feature
        """
        integrated_stats = {}
        
        if 'feature_analysis' not in statistical_report:
            return {'error': 'Formato de relatório estatístico inválido'}
        
        for feature_name, analysis in statistical_report['feature_analysis'].items():
            

            feature_type = analysis['feature_type']
            basic_stats = analysis['basic_statistics']
            type_specific = analysis['type_specific_analysis']
            comparison = analysis.get('comparison_analysis')

            integrated_stats[feature_name] = {
                'feature_type': feature_type,
                'sample_size': basic_stats['sample_size'],
                'null_percentage': basic_stats['null_percentage'],
                'unique_values': basic_stats['unique_values'],
                'type_specific_info': type_specific,
                'comparison_available': comparison is not None,
                'comparison_analysis': comparison,
                'recommended_metrics': self.context_guidelines.get(feature_type, {}).get('primary_metrics', [])
            }
        
        return integrated_stats
    
    def integrate_drift_metrics(self, drift_metrics_results):
        """
        Integra resultados do DriftMetricsCalculator e aplica interpretação científica
        
        Args:
            drift_metrics_results: Output do calculate_metrics_from_report()
        
        Returns:
            dict: Métricas interpretadas por feature
        """
        interpreted_metrics = {}
        
        for feature_name, results in drift_metrics_results.items():
            if feature_name.startswith('_'):  # Skip metadata
                continue
            
            feature_type = results.get('column_type', 'unknown')
            
            if 'error' in results:
                interpreted_metrics[feature_name] = {
                    'status': 'ERROR',
                    'error_message': results['error']
                }
                continue
            
            feature_type = results.get('feature_type', 'unknown')
            metadata = results.get('_metadata', {})
            
            # Interpretar cada métrica calculada baseado em evidências científicas
            metric_interpretations = {}
            
            for metric_name, metric_result in results.items():
                if metric_name in ['feature_type', 'monitoring_priority', 'metric_details', 'sample_size', '_metadata']:
                    continue
                
                if isinstance(metric_result, dict) and 'error' not in metric_result:
                    interpretation = self._interpret_metric_scientifically(
                        metric_name, metric_result, feature_type
                    )
                    metric_interpretations[metric_name] = interpretation
            
            interpreted_metrics[feature_name] = {
                'feature_type': feature_type,
                'monitoring_priority': results.get('monitoring_priority', 'MEDIUM'),
                'sample_size': results.get('sample_size', 0),
                'metrics_calculated': metadata.get('calculated', 0),
                'metric_interpretations': metric_interpretations,
                'context_guidelines': self.context_guidelines.get(feature_type, {})
            }
        
        return interpreted_metrics
    
    def _interpret_metric_scientifically(self, metric_name, metric_result, feature_type):
        """
        Interpreta uma métrica individual baseada em literatura científica
        
        Args:
            metric_name: Nome da métrica
            metric_result: Resultado da métrica
            feature_type: Tipo da feature
        
        Returns:
            dict: Interpretação científica da métrica
        """
        interpretation = {
            'raw_value': None,
            'severity': 'UNKNOWN',
            'confidence': 'UNKNOWN',
            'scientific_interpretation': '',
            'business_recommendation': '',
            'uncertainty_notes': ''
        }
        
        # PSI (Population Stability Index)
        if metric_name == 'psi' and 'psi_value' in metric_result:
            psi_value = metric_result['psi_value']
            
            if not np.isnan(psi_value):

                interpretation['raw_value'] = psi_value
                
                if psi_value < self.thresholds['psi']['stable']:
                    interpretation['severity'] = 'LOW'
                    interpretation['confidence'] = 'HIGH'
                    interpretation['scientific_interpretation'] = f'PSI={psi_value:.3f} < 0.1: Distribuição estável (Narayanan, 2010)'
                    interpretation['business_recommendation'] = 'Continuar monitoramento de rotina'
                elif psi_value < self.thresholds['psi']['moderate']:
                    interpretation['severity'] = 'MEDIUM'
                    interpretation['confidence'] = 'HIGH'
                    interpretation['scientific_interpretation'] = f'0.1 ≤ PSI={psi_value:.3f} < 0.2: Mudança moderada detectada'
                    interpretation['business_recommendation'] = 'Investigar causas e aumentar frequência de monitoramento'
                elif psi_value < self.thresholds['psi']['unstable']:
                    interpretation['severity'] = 'HIGH'
                    interpretation['confidence'] = 'HIGH'
                    interpretation['scientific_interpretation'] = f'0.2 ≤ PSI={psi_value:.3f} < 0.25: Mudança significativa'
                    interpretation['business_recommendation'] = 'Ação corretiva necessária'
                else:
                    interpretation['severity'] = 'CRITICAL'
                    interpretation['confidence'] = 'HIGH'
                    interpretation['scientific_interpretation'] = f'PSI={psi_value:.3f} ≥ 0.25: Mudança drástica na distribuição'
                    interpretation['business_recommendation'] = 'Intervenção imediata necessária'
        
        # KS Test (Kolmogorov-Smirnov)
        elif metric_name == 'ks_test' and 'ks_statistic' in metric_result:
            ks_stat = metric_result['ks_statistic']
            p_value = metric_result.get('p_value', 1.0)
            interpretation['raw_value'] = {'statistic': ks_stat, 'p_value': p_value}
            
            # CORREÇÃO: Avaliar MAGNITUDE do efeito (D) além de significância (p-value)
            # Referência: Cohen (1988), Sawilowsky (2009) - Effect Size Guidelines
            
            if p_value >= self.thresholds['ks_test']['alpha_standard']:  # p >= 0.05
                interpretation['severity'] = 'LOW'
                interpretation['confidence'] = 'HIGH'
                interpretation['scientific_interpretation'] = f'KS-test: p={p_value:.4f} ≥ 0.05, não significativo'
                interpretation['business_recommendation'] = 'Nenhuma ação necessária'
            else:
                # Significativo estatisticamente - avaliar magnitude do efeito
                if ks_stat < 0.05:  # Efeito trivial
                    interpretation['severity'] = 'LOW'
                    interpretation['confidence'] = 'MODERATE'
                    interpretation['scientific_interpretation'] = f'KS-test: p={p_value:.4f} significativo, mas D={ks_stat:.3f} < 0.05 (efeito trivial)'
                    interpretation['business_recommendation'] = 'Significância estatística sem relevância prática'
                    interpretation['uncertainty_notes'] = 'Alta potência estatística detectou mudança mínima (Cohen 1988)'
                elif ks_stat < 0.10:  # Efeito pequeno
                    interpretation['severity'] = 'MEDIUM'
                    interpretation['confidence'] = 'HIGH'
                    interpretation['scientific_interpretation'] = f'KS-test: D={ks_stat:.3f} (pequeno), p={p_value:.4f}'
                    interpretation['business_recommendation'] = 'Mudança pequena mas detectável - monitorar tendência'
                elif ks_stat < 0.20:  # Efeito moderado
                    interpretation['severity'] = 'HIGH'
                    interpretation['confidence'] = 'VERY_HIGH'
                    interpretation['scientific_interpretation'] = f'KS-test: D={ks_stat:.3f} (moderado), p={p_value:.4f}'
                    interpretation['business_recommendation'] = 'Mudança moderada - investigar causas'
                else:  # Efeito grande (D >= 0.20)
                    interpretation['severity'] = 'CRITICAL'
                    interpretation['confidence'] = 'VERY_HIGH'
                    interpretation['scientific_interpretation'] = f'KS-test: D={ks_stat:.3f} (grande), p={p_value:.2e}'
                    interpretation['business_recommendation'] = 'Mudança substancial - ação imediata necessária'
        
        # Chi-squared Test
        elif metric_name == 'chi_square' and 'chi2_statistic' in metric_result:
            chi2_stat = metric_result['chi2_statistic']
            p_value = metric_result.get('p_value', 1.0)
            interpretation['raw_value'] = {'statistic': chi2_stat, 'p_value': p_value}
            
            # Aplicar mesma lógica do KS test para p-values
            if p_value < self.thresholds['chi_square']['alpha_strict']:
                interpretation['severity'] = 'HIGH'
                interpretation['confidence'] = 'VERY_HIGH'
                interpretation['scientific_interpretation'] = f'Chi²-test: p={p_value:.4f} < 0.01, altamente significativo'
                interpretation['business_recommendation'] = 'Forte evidência de mudança nas proporções categóricas'
            elif p_value < self.thresholds['chi_square']['alpha_standard']:
                interpretation['severity'] = 'MEDIUM'
                interpretation['confidence'] = 'HIGH'
                interpretation['scientific_interpretation'] = f'Chi²-test: p={p_value:.4f} < 0.05, estatisticamente significativo'
                interpretation['business_recommendation'] = 'Evidência de mudança nas proporções categóricas'
            else:
                interpretation['severity'] = 'LOW'
                interpretation['confidence'] = 'HIGH'
                interpretation['scientific_interpretation'] = f'Chi²-test: p={p_value:.4f} ≥ 0.05, não significativo'
                interpretation['business_recommendation'] = 'Nenhuma ação necessária'
            
            # Nota especial para dados categóricos
            if feature_type == "categorical":
                interpretation['uncertainty_notes'] = 'Chi-squared é a métrica padrão-ouro para dados categóricos'
        
        # Wasserstein Distance
        elif metric_name == 'wasserstein_distance' and 'wasserstein_distance' in metric_result:
            distance = metric_result['wasserstein_distance']
            interpretation['raw_value'] = distance
            
            # CORREÇÃO: Usar normalização por IQR (mais robusta) quando disponível
            # Referência: Villani (2008) - Optimal Transport Theory
            if 'normalized_by_iqr' in metric_result:
                normalized_dist = metric_result['normalized_by_iqr']
                ref_iqr = metric_result.get('reference_iqr', 1.0)
                
                # Thresholds sobre % do IQR
                if normalized_dist < 0.05:  # < 5% do IQR
                    interpretation['severity'] = 'LOW'
                    interpretation['confidence'] = 'HIGH'
                    interpretation['scientific_interpretation'] = f'Wasserstein={distance:.3f} (~{normalized_dist*100:.1f}% IQR): Diferença mínima'
                    interpretation['business_recommendation'] = 'Variação dentro do esperado'
                elif normalized_dist < 0.15:  # < 15% do IQR
                    interpretation['severity'] = 'MEDIUM'
                    interpretation['confidence'] = 'HIGH'
                    interpretation['scientific_interpretation'] = f'Wasserstein={distance:.3f} (~{normalized_dist*100:.1f}% IQR): Diferença moderada'
                    interpretation['business_recommendation'] = 'Monitorar tendência'
                elif normalized_dist < 0.30:  # < 30% do IQR
                    interpretation['severity'] = 'HIGH'
                    interpretation['confidence'] = 'HIGH'
                    interpretation['scientific_interpretation'] = f'Wasserstein={distance:.3f} (~{normalized_dist*100:.1f}% IQR): Diferença substancial'
                    interpretation['business_recommendation'] = 'Investigar causas da mudança'
                else:  # >= 30% do IQR
                    interpretation['severity'] = 'CRITICAL'
                    interpretation['confidence'] = 'HIGH'
                    interpretation['scientific_interpretation'] = f'Wasserstein={distance:.3f} (~{normalized_dist*100:.1f}% IQR): Grande mudança distribucional'
                    interpretation['business_recommendation'] = 'Ação corretiva imediata'
                
                interpretation['uncertainty_notes'] = f'Distância normalizada por IQR={ref_iqr:.3f} (Earth Mover Distance - Villani 2008)'
            else:
                # Fallback: usar thresholds absolutos (menos robusto)
                if distance < self.thresholds['wasserstein']['low']:
                    interpretation['severity'] = 'LOW'
                    interpretation['confidence'] = 'HIGH'
                    interpretation['scientific_interpretation'] = f'Wasserstein={distance:.3f} < 0.1: Distribuições similares'
                    interpretation['business_recommendation'] = 'Diferença mínima detectada'
                elif distance < self.thresholds['wasserstein']['moderate']:
                    interpretation['severity'] = 'MEDIUM'
                    interpretation['confidence'] = 'HIGH'
                    interpretation['scientific_interpretation'] = f'Wasserstein={distance:.3f} < 0.3: Diferença moderada'
                    interpretation['business_recommendation'] = 'Monitorar tendência'
                elif distance < self.thresholds['wasserstein']['high']:  # BUG FIX: adicionar elif faltante
                    interpretation['severity'] = 'HIGH'
                    interpretation['confidence'] = 'HIGH'
                    interpretation['scientific_interpretation'] = f'Wasserstein={distance:.3f} < 0.5: Diferença substancial'
                    interpretation['business_recommendation'] = 'Investigar causas'
                else:  # >= 0.5
                    interpretation['severity'] = 'CRITICAL'
                    interpretation['confidence'] = 'HIGH'
                    interpretation['scientific_interpretation'] = f'Wasserstein={distance:.3f} ≥ 0.5: Grande diferença distribucional'
                    interpretation['business_recommendation'] = 'Ação imediata'
                
                interpretation['uncertainty_notes'] = 'Usando thresholds absolutos (recomendado: normalizar por escala da feature)'
            
            # Contexto para diferentes tipos de dados
            if feature_type == 'numerical':
                interpretation['uncertainty_notes'] += ' | Earth Mover Distance - adequada para dados contínuos'
            elif feature_type == 'categorical_numeric':
                interpretation['uncertainty_notes'] += ' | Pode capturar ordem em dados categóricos ordinais'
        
        # Hellinger Distance
        elif metric_name == 'hellinger_distance' and 'hellinger_distance' in metric_result:
            distance = metric_result['hellinger_distance']
            interpretation['raw_value'] = distance
            
            if distance < self.thresholds['hellinger']['low']:
                interpretation['severity'] = 'LOW'
                interpretation['confidence'] = 'HIGH'
                interpretation['scientific_interpretation'] = f'Hellinger={distance:.3f} < 0.1: Distribuições muito similares'
                interpretation['business_recommendation'] = 'Diferença negligível'
            elif distance < self.thresholds['hellinger']['moderate']:
                interpretation['severity'] = 'MEDIUM'
                interpretation['confidence'] = 'HIGH'
                interpretation['scientific_interpretation'] = f'Hellinger={distance:.3f}: Diferença moderada'
                interpretation['business_recommendation'] = 'Investigar tendência'
            else:
                interpretation['severity'] = 'HIGH'
                interpretation['confidence'] = 'HIGH'
                interpretation['scientific_interpretation'] = f'Hellinger={distance:.3f} ≥ 0.: Distribuições substancialmente diferentes'
                interpretation['business_recommendation'] = 'Ação corretiva recomendada'
            
            interpretation['uncertainty_notes'] = 'Métrica simétrica e boundada [0,1] - boa para comparação entre features'
        
        # JS Divergence
        elif metric_name == 'js_divergence' and 'js_divergence' in metric_result:
            js_div = metric_result['js_divergence']
            interpretation['raw_value'] = js_div
            
            if js_div < self.thresholds['js_divergence']['low']:
                interpretation['severity'] = 'LOW'
                interpretation['confidence'] = 'HIGH'
                interpretation['scientific_interpretation'] = f'JS-divergence={js_div:.3f} < 0.1: Baixa divergência'
                interpretation['business_recommendation'] = 'Distribuições similares'
            elif js_div < self.thresholds['js_divergence']['moderate']:
                interpretation['severity'] = 'MEDIUM'
                interpretation['confidence'] = 'HIGH'
                interpretation['scientific_interpretation'] = f'JS-divergence={js_div:.3f}: Divergência moderada'
                interpretation['business_recommendation'] = 'Monitorar evolução'
            else:
                interpretation['severity'] = 'HIGH'
                interpretation['confidence'] = 'HIGH'
                interpretation['scientific_interpretation'] = f'JS-divergence={js_div:.3f} ≥ 0.5: Alta divergência'
                interpretation['business_recommendation'] = 'Investigar mudanças'
            
            interpretation['uncertainty_notes'] = 'Versão simétrica da KL-divergence, mais robusta a outliers'
        
        # KL Divergence
        elif metric_name == 'kl_divergence' and 'kl_divergence' in metric_result:
            kl_div = metric_result['kl_divergence']
            interpretation['raw_value'] = kl_div
            
            if kl_div < self.thresholds['kl_divergence']['low']:
                interpretation['severity'] = 'LOW'
                interpretation['confidence'] = 'MODERATE'
                interpretation['scientific_interpretation'] = f'KL-divergence={kl_div:.3f} < 0.1: Baixa divergência'
                interpretation['business_recommendation'] = 'Distribuições similares'
            elif kl_div < self.thresholds['kl_divergence']['moderate']:
                interpretation['severity'] = 'MEDIUM'
                interpretation['confidence'] = 'MODERATE'
                interpretation['scientific_interpretation'] = f'KL-divergence={kl_div:.3f}: Divergência moderada'
                interpretation['business_recommendation'] = 'Monitorar tendência'
            else:
                interpretation['severity'] = 'HIGH'
                interpretation['confidence'] = 'MODERATE'
                interpretation['scientific_interpretation'] = f'KL-divergence={kl_div:.3f} ≥ 1.0: Alta divergência'
                interpretation['business_recommendation'] = 'Investigar mudanças'
            
            interpretation['uncertainty_notes'] = 'Métrica assimétrica, sensível à ordem de comparação'
        
        else:
            interpretation['uncertainty_notes'] = f'Métrica {metric_name} não reconhecida ou dados insuficientes'
        
        return interpretation
    
    def apply_multiple_testing_correction(
        self, 
        p_values_dict: Dict[str, float], 
        method: str = 'fdr_bh',
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Aplica correção de múltiplos testes (Benjamini-Hochberg FDR)
        
        Args:
            p_values_dict: {feature_name: p_value}
            method: 'bonferroni', 'fdr_bh', 'holm'
            alpha: Nível de significância (default: 0.05)
        
        Returns:
            dict com p-values corrigidos e estatísticas
        
        Referência:
            Benjamini & Hochberg (1995) - Controlling the False Discovery Rate
        """
        if not p_values_dict:
            return {'error': 'No p-values provided'}
        
        features = list(p_values_dict.keys())
        p_values = np.array([p_values_dict[f] for f in features])
        
        # Remover NaN/inf
        valid_mask = np.isfinite(p_values)
        if not valid_mask.any():
            return {'error': 'All p-values are NaN or inf'}
        
        features_valid = [f for i, f in enumerate(features) if valid_mask[i]]
        p_values_valid = p_values[valid_mask]
        
        # Aplicar correção
        reject, p_corrected, alphacSidak, alphacBonf = multipletests(
            p_values_valid, 
            alpha=alpha, 
            method=method
        )
        
        # Criar resultado
        corrected_dict = {}
        for i, feature in enumerate(features_valid):
            corrected_dict[feature] = {
                'p_value_raw': float(p_values_valid[i]),
                'p_value_corrected': float(p_corrected[i]),
                'reject_h0': bool(reject[i]),
                'significant': bool(p_corrected[i] < alpha)
            }
        
        # Estatísticas gerais
        n_significant_raw = int(np.sum(p_values_valid < alpha))
        n_significant_corrected = int(np.sum(reject))
        
        return {
            'corrected_p_values': corrected_dict,
            'method': method,
            'alpha': alpha,
            'n_features_tested': len(features_valid),
            'n_significant_raw': n_significant_raw,
            'n_significant_corrected': n_significant_corrected,
            'false_discovery_rate': alpha,  # Para FDR-BH
            'interpretation': self._interpret_correction_results(
                n_significant_raw, n_significant_corrected, method
            )
        }
    
    def _interpret_correction_results(
        self, 
        n_raw: int, 
        n_corrected: int, 
        method: str
    ) -> str:
        """Gera interpretação textual dos resultados da correção"""
        if n_raw == n_corrected:
            return f"Todos os {n_raw} resultados significativos se mantiveram após correção {method}"
        elif n_corrected == 0:
            return f"Nenhum resultado sobreviveu à correção {method} (possíveis falsos positivos)"
        else:
            n_lost = n_raw - n_corrected
            pct_lost = (n_lost / n_raw * 100) if n_raw > 0 else 0
            return (f"Correção {method} eliminou {n_lost} resultados ({pct_lost:.1f}% dos "
                   f"significativos), mantendo {n_corrected} com evidência robusta")
    
    def _extract_p_values_from_metrics(self, drift_metrics_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Extrai p-values de testes estatísticos das métricas calculadas
        
        Args:
            drift_metrics_results: Resultados do DriftMetricsCalculator
        
        Returns:
            Dict mapping 'feature_metric' → p_value
        """
        p_values = {}
        
        for feature_name, results in drift_metrics_results.items():
            if feature_name.startswith('_'):  # Skip metadata
                continue
            
            if 'error' in results:
                continue
            
            # Extrair p-values de testes que os possuem
            # KS Test
            if 'ks_test' in results and isinstance(results['ks_test'], dict):
                p_val = results['ks_test'].get('p_value')
                if p_val is not None and not np.isnan(p_val):
                    p_values[f"{feature_name}_ks_test"] = float(p_val)
            
            # Chi-squared Test
            if 'chi_square' in results and isinstance(results['chi_square'], dict):
                p_val = results['chi_square'].get('p_value')
                if p_val is not None and not np.isnan(p_val):
                    p_values[f"{feature_name}_chi_square"] = float(p_val)
        
        return p_values
    
    def _update_interpretations_with_correction(
        self, 
        interpreted_metrics: Dict[str, Any], 
        correction_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Atualiza interpretações com resultados da correção de múltiplos testes
        
        Args:
            interpreted_metrics: Métricas interpretadas originais
            correction_results: Resultados do apply_multiple_testing_correction
        
        Returns:
            Métricas interpretadas atualizadas
        """
        if not correction_results or 'corrected_p_values' not in correction_results:
            return interpreted_metrics
        
        corrected_p_values = correction_results['corrected_p_values']
        
        for key, correction_data in corrected_p_values.items():
            # Parse: 'feature_name_metric' → feature_name, metric
            parts = key.rsplit('_', 1)
            if len(parts) != 2:
                parts = key.rsplit('_', 2)  # Try 'feature_chi_square'
            
            if len(parts) >= 2:
                metric = parts[-1]
                feature_name = '_'.join(parts[:-1])
            else:
                continue
            
            if feature_name not in interpreted_metrics:
                continue
            
            if 'metric_interpretations' not in interpreted_metrics[feature_name]:
                continue
            
            if metric not in interpreted_metrics[feature_name]['metric_interpretations']:
                continue
            
            # Adicionar informações de correção
            interp = interpreted_metrics[feature_name]['metric_interpretations'][metric]
            interp['multiple_testing'] = {
                'p_value_raw': correction_data['p_value_raw'],
                'p_value_corrected': correction_data['p_value_corrected'],
                'significant_after_correction': correction_data['significant'],
                'correction_method': correction_results['method']
            }
            
            # Atualizar severidade se perdeu significância
            if not correction_data['significant'] and interp.get('severity') in ['HIGH', 'CRITICAL']:
                interp['severity_before_correction'] = interp['severity']
                interp['severity'] = 'LOW'
                interp['confidence'] = 'LOW'
                interp['uncertainty_notes'] = (
                    f"{interp.get('uncertainty_notes', '')} | "
                    f"⚠️ Perdeu significância após correção FDR (possível falso positivo)"
                ).strip(' | ')
        
        return interpreted_metrics
    
    def calculate_model_impact_analysis(
        self, 
        reference_df: pd.DataFrame, 
        current_df: pd.DataFrame, 
        target_column: str,
        max_samples_for_shap_kernel: int = 2000
    ) -> Dict:
        """
        Calcula o impacto do drift no modelo comparando a importância das features
        entre os datasets de referência e atual.
        
        Escolhe dinamicamente entre SHAP e Permutation Importance.
        """
        impact_analysis = {
            'status': 'NOT_EXECUTED',
            'method_chosen': 'None',
            'baseline_importance': None,
            'current_importance': None,
            'impact_summary': None
        }

        if self.model is None:
            impact_analysis['status'] = 'MODEL_NOT_PROVIDED'
            return impact_analysis
            
        try:
            # Preparar os dados
            y_ref = reference_df[target_column]
            X_ref = reference_df.drop(columns=[target_column])
            
            y_curr = current_df[target_column]
            X_curr = current_df.drop(columns=[target_column])

            # Garantir que as colunas estão na mesma ordem
            X_curr = X_curr[X_ref.columns]
            self.feature_names = X_ref.columns.tolist()

            # Lógica de decisão para o método
            is_tree_model = hasattr(self.model, 'feature_importances_') and \
                            any(s in str(type(self.model)).lower() for s in ['forest', 'booster', 'xgboost'])
            
            use_shap = False
            if is_tree_model:
                use_shap = True
                impact_analysis['method_chosen'] = 'SHAP TreeExplainer'
            elif len(X_curr) < max_samples_for_shap_kernel:
                use_shap = True
                impact_analysis['method_chosen'] = 'SHAP KernelExplainer (slow)'
            else:
                use_shap = False
                impact_analysis['method_chosen'] = 'Permutation Importance'

            # Calcular importâncias
            if use_shap:
                explainer = shap.Explainer(self.model, X_ref)
                ref_shap_values = explainer(X_ref)
                curr_shap_values = explainer(X_curr)
                
                # Global Importance (mean absolute SHAP value)
                ref_importance = pd.Series(np.abs(ref_shap_values.values).mean(axis=0), index=self.feature_names)
                curr_importance = pd.Series(np.abs(curr_shap_values.values).mean(axis=0), index=self.feature_names)
            else: # Permutation Importance
                ref_result = permutation_importance(self.model, X_ref, y_ref, n_repeats=10, random_state=42)
                curr_result = permutation_importance(self.model, X_curr, y_curr, n_repeats=10, random_state=42)
                
                ref_importance = pd.Series(ref_result.importances_mean, index=self.feature_names)
                curr_importance = pd.Series(curr_result.importances_mean, index=self.feature_names)
                
            # Normalizar e ordenar
            ref_importance = (ref_importance / ref_importance.sum()).sort_values(ascending=False)
            curr_importance = (curr_importance / curr_importance.sum()).sort_values(ascending=False)
            
            impact_analysis['baseline_importance'] = ref_importance.to_dict()
            impact_analysis['current_importance'] = curr_importance.to_dict()

            # Gerar sumário do impacto
            impact_analysis['impact_summary'] = self._summarize_importance_changes(ref_importance, curr_importance)
            impact_analysis['status'] = 'SUCCESS'

        except Exception as e:
            impact_analysis['status'] = 'ANALYSIS_FAILED'
            impact_analysis['error'] = str(e)
        
        return impact_analysis

    def _summarize_importance_changes(self, ref_importance, curr_importance):
        """Gera insights a partir da mudança na importância das features."""
        summary = {}
        
        ref_rank = pd.Series(range(len(ref_importance)), index=ref_importance.index)
        curr_rank = pd.Series(range(len(curr_importance)), index=curr_importance.index)
        
        rank_change = curr_rank - ref_rank
        
        summary['top_5_rank_climbers'] = rank_change.sort_values(ascending=False).head(5).to_dict()
        summary['top_5_rank_fallers'] = rank_change.sort_values(ascending=True).head(5).to_dict()
        
        importance_change_pct = ((curr_importance - ref_importance) / ref_importance).fillna(0) * 100
        summary['top_5_gainers_pct'] = importance_change_pct.sort_values(ascending=False).head(5).to_dict()
        summary['top_5_losers_pct'] = importance_change_pct.sort_values(ascending=True).head(5).to_dict()
        
        return summary
    
    def generate_report(self, 
                        statistical_report, 
                        drift_metrics_results, 
                        reference_df, 
                        current_df,
                        include_model_impact=False, 
                        target_column=None):
        """
        Gera relatório completo baseado em evidências científicas
        
        Args:
            statistical_report: Output do DatasetAnalyzer
            drift_metrics_results: Output do DriftMetricsCalculator
            reference_df: DataFrame de referência (antes do drift)
            current_df: DataFrame atual (depois do drift)
            include_model_impact: Se deve incluir análise de impacto no modelo
            target_column: Nome da coluna target
        
        Returns:
            dict: Relatório completo estruturado
        """
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'methodology': 'Evidence-based drift analysis',
            'scientific_references': [
                'PSI Thresholds: Narayanan (2010), SAS Institute',
                'KS Test: Massey Jr. (1951), Kolmogorov (1933)',
                'Chi-squared: Pearson (1900), Fisher (1922)',
                'Wasserstein: Kantorovich-Rubinstein theorem',
                'Hellinger: Hellinger (1909), Beran (1977)'
            ]
        }
        
        print("INICIANDO RELATÓRIO DE...")
        print("=" * 70)
        
        # 1. Integrar estatísticas básicas
        print(" 1. Integrando estatísticas básicas do dataset...")
        integrated_stats = self.integrate_dataset_statistics(statistical_report)
        
        # 2. Interpretar métricas de drift
        print(" 2. Interpretando métricas de drift com base científica...")
        interpreted_metrics = self.integrate_drift_metrics(drift_metrics_results)
        
        # 3. Correção de múltiplos testes (Benjamini-Hochberg FDR)
        print(" 3. Aplicando correção de múltiplos testes (Benjamini-Hochberg)...")
        p_values_dict = self._extract_p_values_from_metrics(drift_metrics_results)
        multiple_testing_correction = None
        if p_values_dict:
            multiple_testing_correction = self.apply_multiple_testing_correction(
                p_values_dict, method='fdr_bh', alpha=0.05
            )
            # Atualizar interpretações com p-values corrigidos
            interpreted_metrics = self._update_interpretations_with_correction(
                interpreted_metrics, multiple_testing_correction
            )
        
        # 4. Análise de impacto no modelo (se solicitado)
        model_impact = None
        if include_model_impact:
        
            model_impact = self.calculate_model_impact_analysis(
                reference_df, current_df, target_column
            )
        
        # 5. Compilar relatório final
        report['dataset_statistics'] = integrated_stats
        report['drift_analysis'] = interpreted_metrics
        if multiple_testing_correction:
            report['multiple_testing_correction'] = multiple_testing_correction
        if model_impact:
            report['model_impact_analysis'] = model_impact
        
        # 6. Gerar resumo executivo
        executive_summary = self._generate_executive_summary(
            integrated_stats, interpreted_metrics, multiple_testing_correction
        )
        report['executive_summary'] = executive_summary
        
        print("✅ Relatório completo gerado!")
        print("=" * 70)
        
        return report
    
    def _generate_executive_summary(self, integrated_stats, interpreted_metrics, multiple_testing_correction=None):
        """
        Gera resumo executivo baseado em evidências
        """
        summary = {
            'total_features_analyzed': len(interpreted_metrics),
            'features_with_significant_drift': 0,
            'features_with_significant_drift_corrected': 0,  # NOVO: Após correção FDR
            'high_confidence_findings': 0,
            'requires_investigation': 0,
            'primary_concerns': [],
            'false_positive_warnings': [],
            'features_with_new_categories': 0,
            'features_with_missing_categories': 0,
            'critical_categorical_drift': [],
            'categorical_drift_summary': {
                'total_new_categories': 0,
                'total_missing_categories': 0,
                'severity': 'NONE'
            }
        }

        for feature_name, metrics in interpreted_metrics.items():
            if 'metric_interpretations' not in metrics:
                continue
            
            feature_drift_detected = False
            feature_drift_after_correction = False
            high_confidence_metrics = 0
            
            for metric_name, interpretation in metrics['metric_interpretations'].items():
                # Verificar se tem correção de múltiplos testes
                has_correction = 'multiple_testing' in interpretation
                is_significant_corrected = (
                    interpretation['multiple_testing']['significant_after_correction'] 
                    if has_correction else True
                )
                
                if interpretation['severity'] in ['HIGH', 'CRITICAL']:
                    feature_drift_detected = True
                    
                    # Contar apenas se passou pela correção (se aplicável)
                    if is_significant_corrected:
                        feature_drift_after_correction = True
                    
                    if interpretation['confidence'] in ['HIGH', 'VERY_HIGH']:
                        high_confidence_metrics += 1
                        
                        # Adicionar às preocupações primárias (apenas se significativo após correção)
                        if is_significant_corrected:
                            concern = {
                                'feature': feature_name,
                                'metric': metric_name,
                                'severity': interpretation['severity'],
                                'confidence': interpretation['confidence'],
                                'interpretation': interpretation['scientific_interpretation'],
                                'recommendation': interpretation['business_recommendation']
                            }
                            if has_correction:
                                concern['corrected_p_value'] = interpretation['multiple_testing']['p_value_corrected']
                            summary['primary_concerns'].append(concern)
                
                # Detectar possíveis falsos positivos
                if has_correction and not is_significant_corrected:
                    if interpretation.get('severity_before_correction') in ['HIGH', 'CRITICAL']:
                        summary['false_positive_warnings'].append({
                            'feature': feature_name,
                            'metric': metric_name,
                            'severity_before': interpretation['severity_before_correction'],
                            'p_value_raw': interpretation['multiple_testing']['p_value_raw'],
                            'p_value_corrected': interpretation['multiple_testing']['p_value_corrected'],
                            'message': 'Perdeu significância após correção FDR (provável falso positivo)'
                        })
            
        
            if feature_drift_detected:
                summary['features_with_significant_drift'] += 1
            
            if feature_drift_after_correction:
                summary['features_with_significant_drift_corrected'] += 1
                
            if high_confidence_metrics > 0:
                summary['high_confidence_findings'] += 1
            
            # Features que requerem investigação (múltiplas métricas moderadas)
            moderate_metrics = sum(1 for interp in metrics['metric_interpretations'].values() 
                                 if interp['severity'] == 'MEDIUM')
            if moderate_metrics >= 2:
                summary['requires_investigation'] += 1
            
            if feature_name in integrated_stats:
                feature_stats = integrated_stats[feature_name]
                comparison = feature_stats.get('comparison_analysis')
                
                if comparison and 'categorical_changes' in comparison:
                    cat_changes = comparison['categorical_changes']
                    
                    # Verificar novas categorias
                    if cat_changes.get('new_categories'):
                        new_cats = cat_changes['new_categories']
                        n_new = len(new_cats)
                        
                        summary['features_with_new_categories'] += 1
                        summary['categorical_drift_summary']['total_new_categories'] += n_new
                        
                        # Determinar severidade baseada em quantidade
                        if n_new >= 5:
                            severity = 'CRITICAL'
                            risk_level = 'IMMEDIATE ACTION REQUIRED'
                        elif n_new >= 3:
                            severity = 'HIGH'
                            risk_level = 'High risk of model failure'
                        elif n_new >= 2:
                            severity = 'MEDIUM'
                            risk_level = 'Moderate risk - requires investigation'
                        else:
                            severity = 'LOW'
                            risk_level = 'Low risk - monitor closely'
                        
                        summary['critical_categorical_drift'].append({
                            'feature': feature_name,
                            'drift_type': 'NEW_CATEGORIES',
                            'new_categories': new_cats,
                            'n_new': n_new,
                            'severity': severity,
                            'risk': f'Model may fail with unseen category values: {new_cats[:3]}{"..." if n_new > 3 else ""}',
                            'risk_level': risk_level,
                            'action_required': 'IMMEDIATE' if severity == 'CRITICAL' else 'HIGH_PRIORITY',
                            'recommended_actions': [
                                'Update feature encoding to handle new categories',
                                'Retrain model with new category representation' if n_new >= 3 else 'Map new categories to "OTHER" bucket',
                                'Validate model performance on new data',
                                'Monitor prediction quality for affected feature'
                            ]
                        })
                    
                    # Verificar categorias perdidas (menos crítico mas importante)
                    if cat_changes.get('missing_categories'):
                        missing_cats = cat_changes['missing_categories']
                        n_missing = len(missing_cats)
                        
                        summary['features_with_missing_categories'] += 1
                        summary['categorical_drift_summary']['total_missing_categories'] += n_missing
                        
                        summary['critical_categorical_drift'].append({
                            'feature': feature_name,
                            'drift_type': 'MISSING_CATEGORIES',
                            'missing_categories': missing_cats,
                            'n_missing': n_missing,
                            'severity': 'MEDIUM' if n_missing >= 3 else 'LOW',
                            'risk': f'Data distribution shifted - {n_missing} categories disappeared',
                            'risk_level': 'Data quality concern',
                            'action_required': 'INVESTIGATION',
                            'recommended_actions': [
                                'Investigate why categories disappeared',
                                'Check data collection pipeline',
                                'Verify if change is expected (business logic)',
                                'Consider retraining if distribution changed significantly'
                            ]
                        })
        
        #Determinar severidade globas de drift categorico
        if summary['critical_categorical_drift']:
            max_severity = max(
                alert['severity'] 
                for alert in summary['critical_categorical_drift']
            )
            summary['categorical_drift_summary']['severity'] = max_severity
        
        # Adicionar estatísticas da correção se disponível
        if multiple_testing_correction:
            summary['multiple_testing_stats'] = {
                'method': multiple_testing_correction['method'],
                'n_tests': multiple_testing_correction['n_features_tested'],
                'n_significant_before': multiple_testing_correction['n_significant_raw'],
                'n_significant_after': multiple_testing_correction['n_significant_corrected'],
                'interpretation': multiple_testing_correction['interpretation']
            }
        
        return summary
    
    def print_report(self, report, full_report_json:bool=False):
        """
        Imprime relatório formatado
        """
        print("\n" + "=" * 80)
        print(" RELATÓRIO DE DRIFT")

        if full_report_json:
            print(f"Printing Full Report as JSON:\n{report}")
            return json.dumps(report, indent=4, ensure_ascii=False)
        else:
            # Executive Summary
            summary = report['executive_summary']

            if summary.get('critical_categorical_drift'):
                print("ALERTAS CRÍTICOS: DRIFT CATEGÓRICO DETECTADO")
                
                cat_summary = summary['categorical_drift_summary']
                print(f"\nRESUMO:")
                print(f"   • Features com NOVAS categorias: {summary['features_with_new_categories']}")
                print(f"   • Features com categorias PERDIDAS: {summary['features_with_missing_categories']}")
                print(f"   • Total de novas categorias: {cat_summary['total_new_categories']}")
                print(f"   • Total de categorias perdidas: {cat_summary['total_missing_categories']}")
                print(f"   • Severidade global: {cat_summary['severity']}")
                
                print(f"\n  🔍 DETALHES POR FEATURE:")
                for alert in summary['critical_categorical_drift']:
                    icon = '❌' if alert['severity'] in ['CRITICAL', 'HIGH'] else '⚠️'
                    print(f"\n   {icon} {alert['feature']} ({alert['drift_type']}):")
                    print(f"      • Severidade: {alert['severity']}")
                    print(f"      • Risco: {alert['risk']}")
                    print(f"      • Nível de Risco: {alert['risk_level']}")
                    print(f"      • Ação Requerida: {alert['action_required']}")
                    
                    if alert['drift_type'] == 'NEW_CATEGORIES':
                        print(f"      • Novas categorias ({alert['n_new']}): {', '.join(map(str, alert['new_categories'][:5]))}")
                        if alert['n_new'] > 5:
                            print(f"        ... e mais {alert['n_new'] - 5} categorias")
                    elif alert['drift_type'] == 'MISSING_CATEGORIES':
                        print(f"      • Categorias perdidas ({alert['n_missing']}): {', '.join(map(str, alert['missing_categories'][:5]))}")
                        if alert['n_missing'] > 5:
                            print(f"        ... e mais {alert['n_missing'] - 5} categorias")
                    
                    print(f"      • Ações Recomendadas:")
                    for i, action in enumerate(alert['recommended_actions'], 1):
                        print(f"        {i}. {action}")
                
                print("\n" + "=" * 80)
            

            print(f"\n  RESUMO EXECUTIVO:")
            print(f"   • Features analisadas: {summary['total_features_analyzed']}")
            print(f"   • Features com drift significativo (raw): {summary['features_with_significant_drift']}")
            
            # Mostrar correção se disponível
            if 'multiple_testing_stats' in summary:
                mt_stats = summary['multiple_testing_stats']
                print(f"   • Features com drift (após {mt_stats['method'].upper()}): {summary['features_with_significant_drift_corrected']}")
                print(f"   • Testes realizados: {mt_stats['n_tests']}")
                print(f"   • Significativos antes: {mt_stats['n_significant_before']}")
                print(f"   • Significativos depois: {mt_stats['n_significant_after']}")
            
            print(f"   • Achados de alta confiança: {summary['high_confidence_findings']}")
            print(f"   • Requerem investigação: {summary['requires_investigation']}")
            
            if summary.get('features_with_new_categories', 0) > 0:
                print(f"\n  ⚠️  DRIFT CATEGÓRICO:")
                print(f"   • Features com novas categorias: {summary['features_with_new_categories']} 🔴")
                print(f"   • Features com categorias perdidas: {summary['features_with_missing_categories']}")
            
            
            # Avisos de Falsos Positivos
            if summary.get('false_positive_warnings'):
                print(f"\n  ⚠️ AVISOS DE POSSÍVEIS FALSOS POSITIVOS:")
                for warning in summary['false_positive_warnings'][:3]:  # Top 3
                    print(f"   • {warning['feature']} ({warning['metric']})")
                    print(f"     - Severidade original: {warning['severity_before']}")
                    print(f"     - p-value raw: {warning['p_value_raw']:.4f}")
                    print(f"     - p-value corrigido: {warning['p_value_corrected']:.4f}")
                    print(f"     - {warning['message']}")
            
            # Primary Concerns
            if summary['primary_concerns']:
                print(f"\n  PREOCUPAÇÕES PRIMÁRIAS (ALTA CONFIANÇA):")
                for concern in summary['primary_concerns'][:5]:  # Top 5
                    print(f"      {concern['feature']} ({concern['metric']}):")
                    print(f"      • Severidade: {concern['severity']}")
                    print(f"      • Confiança: {concern['confidence']}")
                    if 'corrected_p_value' in concern:
                        print(f"      • p-value corrigido: {concern['corrected_p_value']:.4f}")
                    print(f"      • Interpretação: {concern['interpretation']}")
                    print(f"      • Recomendação: {concern['recommendation']}")
                    print()
            
            # Estatísticas de Correção Múltipla
            if 'multiple_testing_correction' in report:
                mt_corr = report['multiple_testing_correction']
                print(f"\n  📊 CORREÇÃO DE MÚLTIPLOS TESTES:")
                print(f"   • Método: {mt_corr['method'].upper()} (Benjamini-Hochberg)")
                print(f"   • {mt_corr['interpretation']}")
                print(f"   • FDR (False Discovery Rate): {mt_corr['false_discovery_rate']:.2f}")
            
            # Scientific References
            print(f"\n  REFERÊNCIAS CIENTÍFICAS:")
            for ref in report['scientific_references']:
                print(f"   • {ref}")
            print(f"   • Multiple Testing: Benjamini & Hochberg (1995)")
            
            # Model Impact (if available)
            if 'model_impact_analysis' in report:
                impact = report['model_impact_analysis']
                print(f"\n ANÁLISE DE IMPACTO NO MODELO:")
                print(f"   • Status: {impact['status']}")
                if 'limitations' in impact:
                    print(f"   • Limitações:")
                    for limitation in impact['limitations']:
                        print(f"     - {limitation}")
            
            print("\n" + "=" * 80)
