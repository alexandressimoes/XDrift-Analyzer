"""
🚀 Melhorias de Código para XAdapt-Drift POC
============================================

Código pronto para implementação das melhorias críticas identificadas.
Copie e cole os métodos nas respectivas classes.

Desenvolvedor: Python Senior & ML Engineer
Data: 02/10/2025
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from statsmodels.stats.multitest import multipletests



# ========================================================================
# 2. BINNING UNIFICADO PARA PSI (drift_metrics_calculator.py)
# ========================================================================

class DriftMetricsCalculatorEnhanced:
    """Versão com binning consistente para todas as métricas"""
    
    def psi_unified(
        self, 
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
            
            # Substituir zeros por epsilon GLOBAL
            ref_prop = np.where(ref_prop == 0, self.EPSILON, ref_prop)
            curr_prop = np.where(curr_prop == 0, self.EPSILON, curr_prop)
            
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


# ========================================================================
# 3. EPSILON PADRONIZADO (drift_metrics_calculator.py)
# ========================================================================

class DriftMetricsCalculatorStandardized:
    """Versão com epsilon global padronizado"""
    
    # Constante global para evitar divisão por zero
    EPSILON = 1e-10  # IEEE 754 double precision: ~2.22e-16
    
    def __init__(self, feature_names=None, default_bins='auto'):
        self.feature_names = feature_names
        self.default_bins = default_bins
        self.metric_methods = {...}  # mantém existente
    
    def _apply_epsilon(self, probabilities: np.ndarray) -> np.ndarray:
        """Método centralizado para aplicar epsilon"""
        return np.where(probabilities == 0, self.EPSILON, probabilities)
    
    def psi(self, reference, current, column_type, bins=None):
        """PSI com epsilon padronizado"""
        # ... código existente ...
        
        # SUBSTITUIR:
        # ref_prop = np.where(ref_prop == 0, 1e-7, ref_prop)
        # curr_prop = np.where(curr_prop == 0, 1e-7, curr_prop)
        
        # POR:
        ref_prop = self._apply_epsilon(ref_prop)
        curr_prop = self._apply_epsilon(curr_prop)
        
        psi_value = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
        return {'psi_value': float(psi_value), ...}
    
    def kl_divergence(self, reference, current, column_type, bins=None):
        """KL Divergence com epsilon padronizado"""
        # ... código existente ...
        
        # SUBSTITUIR linhas com 1e-10 inconsistente
        ref_prob = self._apply_epsilon(ref_prob)
        curr_prob = self._apply_epsilon(curr_prob)
        
        kl_div = np.sum(curr_prob * np.log(curr_prob / ref_prob))
        return {'kl_divergence': float(kl_div), ...}
    
    def jensen_shannon_divergence(self, reference, current, column_type, bins=None):
        """JS Divergence com epsilon padronizado"""
        # ... código existente ...
        
        ref_prob = self._apply_epsilon(ref_prob)
        curr_prob = self._apply_epsilon(curr_prob)
        
        js_div = jensenshannon(ref_prob, curr_prob) ** 2
        return {'js_divergence': float(js_div), ...}


# ========================================================================
# 4. KS-TEST COM MAGNITUDE DO EFEITO (drift_report_generator.py)
# ========================================================================

class KSTestEnhancedInterpretation:
    """Interpretação correta do KS-test (significância + magnitude)"""
    
    def _interpret_ks_test_scientifically(
        self, 
        ks_stat: float, 
        p_value: float,
        sample_size_ref: int,
        sample_size_curr: int
    ) -> Dict[str, Any]:
        """
        Interpreta KS-test considerando:
        1. Significância estatística (p-value)
        2. Magnitude do efeito (D statistic)
        3. Tamanho amostral (potência)
        
        Referências:
        - Cohen (1988): Effect size guidelines
        - Sawilowsky (2009): Expanded effect sizes
        - Sullivan & Feinn (2012): "Significance ≠ Importance"
        """
        interpretation = {
            'raw_value': {'statistic': ks_stat, 'p_value': p_value},
            'sample_sizes': {'reference': sample_size_ref, 'current': sample_size_curr}
        }
        
        # Etapa 1: Verificar significância estatística
        is_significant = p_value < 0.05
        
        if not is_significant:
            # p >= 0.05: Não significativo
            interpretation.update({
                'severity': 'LOW',
                'confidence': 'HIGH',
                'statistical_significance': 'not_significant',
                'scientific_interpretation': (
                    f'KS-test: p={p_value:.4f} ≥ 0.05, não há evidência estatística '
                    f'de diferença entre distribuições'
                ),
                'business_recommendation': 'Nenhuma ação necessária',
                'effect_size_category': 'none'
            })
            return interpretation
        
        # Etapa 2: Avaliar MAGNITUDE do efeito (D statistic)
        # Thresholds baseados em Sawilowsky (2009)
        effect_thresholds = {
            'trivial': 0.05,    # < 0.05
            'small': 0.10,      # 0.05 - 0.10
            'moderate': 0.20,   # 0.10 - 0.20
            'large': 0.30       # >= 0.30 (ajustado de 0.20 para evitar over-sensitivity)
        }
        
        if ks_stat < effect_thresholds['trivial']:
            # Efeito trivial (significativo mas irrelevante na prática)
            interpretation.update({
                'severity': 'LOW',
                'confidence': 'MODERATE',
                'statistical_significance': 'significant',
                'effect_size_category': 'trivial',
                'scientific_interpretation': (
                    f'KS-test: p={p_value:.4f} significativo, mas D={ks_stat:.3f} < 0.05 '
                    f'indica efeito trivial (Cohen 1988). Alta potência estatística detectou '
                    f'diferença mínima sem relevância prática.'
                ),
                'business_recommendation': (
                    'Diferença estatisticamente significativa mas praticamente irrelevante. '
                    'Considere aumentar threshold ou reduzir tamanho amostral.'
                ),
                'uncertainty_notes': (
                    f'Tamanho amostral grande (n_ref={sample_size_ref}, n_curr={sample_size_curr}) '
                    f'aumenta potência e detecta efeitos mínimos'
                )
            })
        
        elif ks_stat < effect_thresholds['small']:
            # Efeito pequeno
            interpretation.update({
                'severity': 'MEDIUM',
                'confidence': 'HIGH',
                'statistical_significance': 'significant',
                'effect_size_category': 'small',
                'scientific_interpretation': (
                    f'KS-test: D={ks_stat:.3f} (pequeno), p={p_value:.4f}. '
                    f'Mudança detectável seguindo Sawilowsky (2009): 0.05 ≤ D < 0.10'
                ),
                'business_recommendation': (
                    'Mudança pequena mas detectável - monitorar tendência ao longo do tempo'
                )
            })
        
        elif ks_stat < effect_thresholds['moderate']:
            # Efeito moderado
            interpretation.update({
                'severity': 'HIGH',
                'confidence': 'VERY_HIGH',
                'statistical_significance': 'significant',
                'effect_size_category': 'moderate',
                'scientific_interpretation': (
                    f'KS-test: D={ks_stat:.3f} (moderado), p={p_value:.4f}. '
                    f'Evidência forte de mudança distribucional (0.10 ≤ D < 0.20)'
                ),
                'business_recommendation': (
                    'Mudança moderada - investigar causas e avaliar impacto no modelo'
                )
            })
        
        else:
            # Efeito grande
            interpretation.update({
                'severity': 'CRITICAL',
                'confidence': 'VERY_HIGH',
                'statistical_significance': 'significant',
                'effect_size_category': 'large',
                'scientific_interpretation': (
                    f'KS-test: D={ks_stat:.3f} (grande), p={p_value:.2e}. '
                    f'Mudança substancial detectada (D ≥ {effect_thresholds["moderate"]})'
                ),
                'business_recommendation': (
                    'Mudança drástica - ação imediata necessária (retreinamento, '
                    'recalibração ou investigação de data quality issues)'
                )
            })
        
        # Etapa 3: Calcular potência estatística (aproximação)
        interpretation['power_analysis'] = self._estimate_ks_power(
            ks_stat, sample_size_ref, sample_size_curr
        )
        
        return interpretation
    
    def _estimate_ks_power(
        self, 
        ks_stat: float, 
        n_ref: int, 
        n_curr: int, 
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Estima potência do KS-test (aproximação via teoria assintótica)
        
        Fórmula aproximada: Power ≈ Φ(z_effect - z_alpha)
        onde z_effect = D * sqrt(n_ref * n_curr / (n_ref + n_curr))
        """
        n_effective = (n_ref * n_curr) / (n_ref + n_curr)
        z_effect = ks_stat * np.sqrt(n_effective)
        z_alpha = stats.norm.ppf(1 - alpha / 2)  # two-sided
        
        power_approx = stats.norm.cdf(z_effect - z_alpha)
        
        return {
            'estimated_power': float(power_approx),
            'interpretation': 'high' if power_approx > 0.8 else 'moderate' if power_approx > 0.5 else 'low',
            'recommendation': (
                'Potência adequada' if power_approx > 0.8 else 
                f'Potência baixa ({power_approx:.2f}) - considere aumentar amostra'
            ),
            'method': 'asymptotic_approximation'
        }


# ========================================================================
# 5. THRESHOLDS ADAPTATIVOS (drift_report_generator.py)
# ========================================================================

class AdaptiveThresholdManager:
    """Gerencia thresholds adaptativos baseados em contexto"""
    
    def __init__(self, base_thresholds: Dict[str, Dict], criticality: str = 'medium'):
        """
        Args:
            base_thresholds: Thresholds padrão (formato atual)
            criticality: 'low', 'medium', 'high' (nível de criticidade do sistema)
        """
        self.base_thresholds = base_thresholds
        self.criticality = criticality
        self.criticality_multipliers = {
            'low': 1.5,      # Mais tolerante
            'medium': 1.0,   # Padrão
            'high': 0.7      # Mais rigoroso
        }
    
    def get_adaptive_threshold(
        self, 
        metric_name: str,
        severity_level: str,  # 'low', 'moderate', 'high'
        context: Dict[str, Any]
    ) -> float:
        """
        Retorna threshold ajustado ao contexto
        
        Args:
            metric_name: 'psi', 'wasserstein', etc.
            severity_level: 'low', 'moderate', 'high'
            context: {
                'sample_size': int,
                'historical_volatility': float (std de valores históricos),
                'feature_importance': float (0-1, se disponível)
            }
        
        Returns:
            float: Threshold ajustado
        """
        # Threshold base
        base = self.base_thresholds.get(metric_name, {}).get(severity_level, 0.1)
        
        # Ajuste 1: Tamanho amostral
        sample_adj = self._sample_size_adjustment(context.get('sample_size', 1000))
        
        # Ajuste 2: Criticidade do sistema
        crit_adj = self.criticality_multipliers[self.criticality]
        
        # Ajuste 3: Volatilidade histórica
        hist_adj = self._historical_volatility_adjustment(
            context.get('historical_volatility', 0.0)
        )
        
        # Ajuste 4: Importância da feature (se disponível)
        importance_adj = self._feature_importance_adjustment(
            context.get('feature_importance', 0.5)
        )
        
        # Combinação multiplicativa (pode mudar para aditiva se necessário)
        adjusted = base * sample_adj * crit_adj * hist_adj * importance_adj
        
        return float(np.clip(adjusted, base * 0.5, base * 2.0))  # Limitar variação
    
    def _sample_size_adjustment(self, n: int) -> float:
        """
        Ajusta threshold baseado em potência estatística
        
        Lógica: Amostras grandes detectam efeitos pequenos → relaxar threshold
        """
        if n > 10000:
            return 1.3  # Relaxar 30%
        elif n > 5000:
            return 1.15
        elif n < 1000:
            return 0.9  # Apertar 10%
        elif n < 500:
            return 0.8
        return 1.0
    
    def _historical_volatility_adjustment(self, volatility: float) -> float:
        """
        Ajusta baseado em volatilidade histórica da feature
        
        Lógica: Features voláteis têm drift "natural" → relaxar threshold
        """
        if volatility > 0.2:  # Alta volatilidade
            return 1.3
        elif volatility > 0.1:  # Moderada
            return 1.15
        elif volatility < 0.05:  # Muito estável
            return 0.85
        return 1.0
    
    def _feature_importance_adjustment(self, importance: float) -> float:
        """
        Ajusta baseado em importância da feature no modelo
        
        Lógica: Features importantes requerem monitoramento mais rigoroso
        """
        if importance > 0.8:  # Muito importante
            return 0.8  # Apertar 20%
        elif importance > 0.5:  # Moderadamente importante
            return 0.9
        elif importance < 0.2:  # Pouco importante
            return 1.2
        return 1.0
    
    def explain_threshold_adjustment(
        self, 
        base: float, 
        adjusted: float, 
        context: Dict[str, Any]
    ) -> str:
        """Gera explicação textual do ajuste"""
        factors = []
        
        if context.get('sample_size', 1000) > 5000:
            factors.append(f"amostra grande (n={context['sample_size']}) → +15-30%")
        
        if context.get('historical_volatility', 0) > 0.15:
            factors.append(f"alta volatilidade histórica ({context['historical_volatility']:.2f}) → +15-30%")
        
        if context.get('feature_importance', 0.5) > 0.7:
            factors.append(f"feature crítica (importance={context['feature_importance']:.2f}) → -10-20%")
        
        if self.criticality == 'high':
            factors.append("sistema crítico → -30%")
        
        pct_change = ((adjusted - base) / base) * 100
        
        return (f"Threshold ajustado de {base:.3f} → {adjusted:.3f} ({pct_change:+.1f}%). "
                f"Fatores: {', '.join(factors) if factors else 'nenhum'}")


# ========================================================================
# 6. INTERVALOS DE CONFIANÇA VIA BOOTSTRAP (drift_metrics_calculator.py)
# ========================================================================

class BootstrapConfidenceIntervals:
    """Calcula intervalos de confiança via bootstrap para métricas"""
    
    def kolmogorov_smirnov_with_ci(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        KS-test com intervalo de confiança via bootstrap
        
        Args:
            reference: Dados de referência
            current: Dados atuais
            n_bootstrap: Número de reamostragens (default: 1000)
            confidence_level: Nível de confiança (default: 0.95)
            random_state: Seed para reprodutibilidade
        
        Returns:
            dict com estatística KS, p-value e IC
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Cálculo observado
        ks_stat_obs, p_value = stats.ks_2samp(reference, current)
        
        # Bootstrap
        ks_bootstrap = []
        p_bootstrap = []
        
        for _ in range(n_bootstrap):
            # Reamostrar com reposição
            ref_boot = np.random.choice(reference, size=len(reference), replace=True)
            curr_boot = np.random.choice(current, size=len(current), replace=True)
            
            # Calcular KS
            ks_boot, p_boot = stats.ks_2samp(ref_boot, curr_boot)
            ks_bootstrap.append(ks_boot)
            p_bootstrap.append(p_boot)
        
        ks_bootstrap = np.array(ks_bootstrap)
        p_bootstrap = np.array(p_bootstrap)
        
        # Calcular percentis
        alpha = 1 - confidence_level
        ci_lower_ks = np.percentile(ks_bootstrap, alpha/2 * 100)
        ci_upper_ks = np.percentile(ks_bootstrap, (1 - alpha/2) * 100)
        
        # Métricas adicionais
        ks_std = np.std(ks_bootstrap)
        ks_bias = np.mean(ks_bootstrap) - ks_stat_obs
        
        return {
            'ks_statistic': float(ks_stat_obs),
            'p_value': float(p_value),
            'confidence_interval': {
                'level': confidence_level,
                'lower': float(ci_lower_ks),
                'upper': float(ci_upper_ks),
                'method': 'percentile_bootstrap'
            },
            'bootstrap_metrics': {
                'n_resamples': n_bootstrap,
                'std_error': float(ks_std),
                'bias': float(ks_bias),
                'mean_bootstrap': float(np.mean(ks_bootstrap))
            },
            'interpretation': self._interpret_ci(ks_stat_obs, ci_lower_ks, ci_upper_ks)
        }
    
    def _interpret_ci(self, point_estimate: float, ci_lower: float, ci_upper: float) -> str:
        """Interpreta intervalo de confiança"""
        width = ci_upper - ci_lower
        relative_width = width / point_estimate if point_estimate > 0 else float('inf')
        
        if relative_width < 0.2:
            precision = "alta precisão"
        elif relative_width < 0.5:
            precision = "precisão moderada"
        else:
            precision = "baixa precisão (considere aumentar amostra)"
        
        # Verificar se inclui zero (para H0: não há diferença)
        includes_zero = ci_lower <= 0 <= ci_upper
        
        if includes_zero:
            significance = "IC inclui zero - diferença pode não ser significativa"
        else:
            significance = "IC não inclui zero - diferença robusta"
        
        return (f"D={point_estimate:.3f}, IC95%=[{ci_lower:.3f}, {ci_upper:.3f}] "
                f"({precision}, {significance})")
    
    def psi_with_ci(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """PSI com intervalo de confiança via bootstrap"""
        if random_state is not None:
            np.random.seed(random_state)
        
        # Função helper para calcular PSI
        def calc_psi(ref, curr, bins):
            ref_hist, bin_edges = np.histogram(ref, bins=bins)
            curr_hist, _ = np.histogram(curr, bins=bin_edges)
            
            ref_prop = ref_hist / len(ref)
            curr_prop = curr_hist / len(curr)
            
            ref_prop = np.where(ref_prop == 0, 1e-10, ref_prop)
            curr_prop = np.where(curr_prop == 0, 1e-10, curr_prop)
            
            return np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
        
        # PSI observado
        psi_obs = calc_psi(reference, current, bins)
        
        # Bootstrap
        psi_bootstrap = []
        for _ in range(n_bootstrap):
            ref_boot = np.random.choice(reference, size=len(reference), replace=True)
            curr_boot = np.random.choice(current, size=len(current), replace=True)
            psi_boot = calc_psi(ref_boot, curr_boot, bins)
            psi_bootstrap.append(psi_boot)
        
        psi_bootstrap = np.array(psi_bootstrap)
        
        # IC
        alpha = 1 - confidence_level
        ci_lower = np.percentile(psi_bootstrap, alpha/2 * 100)
        ci_upper = np.percentile(psi_bootstrap, (1 - alpha/2) * 100)
        
        return {
            'psi_value': float(psi_obs),
            'confidence_interval': {
                'level': confidence_level,
                'lower': float(ci_lower),
                'upper': float(ci_upper)
            },
            'bootstrap_metrics': {
                'n_resamples': n_bootstrap,
                'std_error': float(np.std(psi_bootstrap)),
                'bias': float(np.mean(psi_bootstrap) - psi_obs)
            },
            'bins_used': bins,
            'regulatory_compliant': True
        }


# ========================================================================
# 7. DETECÇÃO DE TIPOS ORDINAIS (dataset_analyser.py)
# ========================================================================

class OrdinalDetection:
    """Detecta automaticamente features categóricas ordinais"""
    
    # Padrões conhecidos de categorias ordenadas
    ORDERED_PATTERNS = [
        # Intensidade
        ['very_low', 'low', 'medium', 'high', 'very_high'],
        ['muito_baixo', 'baixo', 'medio', 'alto', 'muito_alto'],
        ['none', 'low', 'medium', 'high'],
        ['small', 'medium', 'large'],
        ['pequeno', 'medio', 'grande'],
        
        # Qualidade
        ['poor', 'fair', 'good', 'excellent'],
        ['ruim', 'regular', 'bom', 'excelente'],
        ['bad', 'neutral', 'good'],
        
        # Frequência
        ['never', 'rarely', 'sometimes', 'often', 'always'],
        ['nunca', 'raramente', 'as_vezes', 'frequentemente', 'sempre'],
        
        # Concordância
        ['strongly_disagree', 'disagree', 'neutral', 'agree', 'strongly_agree'],
        
        # Educação
        ['elementary', 'high_school', 'bachelor', 'master', 'phd'],
        ['fundamental', 'medio', 'superior', 'pos_graduacao']
    ]
    
    @staticmethod
    def detect_ordinal_feature(series: pd.Series) -> Tuple[bool, Optional[List[str]]]:
        """
        Detecta se uma série categórica tem ordem natural
        
        Args:
            series: Pandas Series categórica
        
        Returns:
            (is_ordinal, ordered_categories)
        """
        if not isinstance(series.dtype, pd.CategoricalDtype):
            return False, None
        
        unique_vals = set([str(v).lower().strip() for v in series.dropna().unique()])
        
        # Verificar padrões conhecidos
        for pattern in OrdinalDetection.ORDERED_PATTERNS:
            pattern_set = set([str(p).lower() for p in pattern])
            
            if unique_vals.issubset(pattern_set):
                # Retornar ordem preservando case original
                ordered = [v for v in pattern if str(v).lower() in unique_vals]
                return True, ordered
        
        # Detectar padrões numéricos (level_1, level_2, ...)
        if OrdinalDetection._is_numeric_level_pattern(unique_vals):
            ordered = OrdinalDetection._extract_numeric_order(series.unique())
            return True, ordered
        
        return False, None
    
    @staticmethod
    def _is_numeric_level_pattern(values: set) -> bool:
        """Detecta padrões como level_1, level_2, tier_1, tier_2"""
        import re
        pattern = re.compile(r'^(level|tier|step|stage|grade)_\d+$', re.IGNORECASE)
        return all(pattern.match(v) for v in values)
    
    @staticmethod
    def _extract_numeric_order(unique_values: np.ndarray) -> List[str]:
        """Extrai ordem numérica de padrões como level_1, level_2"""
        import re
        
        def extract_number(s):
            match = re.search(r'_(\d+)$', str(s))
            return int(match.group(1)) if match else 0
        
        sorted_vals = sorted(unique_values, key=extract_number)
        return list(sorted_vals)
    
    @staticmethod
    def suggest_ordinal_encoding(
        series: pd.Series, 
        ordered_categories: List[str]
    ) -> pd.Series:
        """
        Converte categórica ordinal em numérica preservando ordem
        
        Args:
            series: Série categórica original
            ordered_categories: Lista ordenada de categorias
        
        Returns:
            pd.Series com encoding ordinal (0, 1, 2, ...)
        """
        # Criar mapping
        ordinal_map = {cat: i for i, cat in enumerate(ordered_categories)}
        
        # Aplicar encoding
        encoded = series.map(ordinal_map)
        
        return encoded


# ========================================================================
# 8. EXEMPLO DE INTEGRAÇÃO COMPLETA
# ========================================================================

def example_integration():
    """Exemplo de como usar as melhorias em conjunto"""
    
    # Simular dados
    np.random.seed(42)
    reference = np.random.normal(0, 1, 1000)
    current = np.random.normal(0.3, 1, 1000)  # Drift moderado
    
    # 1. Calcular KS com IC
    bootstrap = BootstrapConfidenceIntervals()
    ks_result = bootstrap.kolmogorov_smirnov_with_ci(
        reference, current, 
        n_bootstrap=1000, 
        confidence_level=0.95,
        random_state=42
    )
    
    print("=== KS Test com IC ===")
    print(f"D = {ks_result['ks_statistic']:.3f}")
    print(f"IC 95% = [{ks_result['confidence_interval']['lower']:.3f}, "
          f"{ks_result['confidence_interval']['upper']:.3f}]")
    print(f"Interpretação: {ks_result['interpretation']}")
    
    # 2. Interpretação com magnitude do efeito
    ks_interp = KSTestEnhancedInterpretation()
    interpretation = ks_interp._interpret_ks_test_scientifically(
        ks_stat=ks_result['ks_statistic'],
        p_value=ks_result['p_value'],
        sample_size_ref=len(reference),
        sample_size_curr=len(current)
    )
    
    print("\n=== Interpretação Científica ===")
    print(f"Severidade: {interpretation['severity']}")
    print(f"Categoria de Efeito: {interpretation['effect_size_category']}")
    print(f"Interpretação: {interpretation['scientific_interpretation']}")
    print(f"Recomendação: {interpretation['business_recommendation']}")
    
    # 3. Threshold adaptativo
    threshold_mgr = AdaptiveThresholdManager(
        base_thresholds={'ks_test': {'moderate': 0.10}},
        criticality='high'
    )
    
    context = {
        'sample_size': len(reference),
        'historical_volatility': 0.15,
        'feature_importance': 0.85
    }
    
    adaptive_threshold = threshold_mgr.get_adaptive_threshold(
        'ks_test', 'moderate', context
    )
    
    print(f"\n=== Threshold Adaptativo ===")
    print(f"Threshold base: 0.10")
    print(f"Threshold ajustado: {adaptive_threshold:.3f}")
    print(threshold_mgr.explain_threshold_adjustment(0.10, adaptive_threshold, context))
    
    # 4. Correção de múltiplos testes (simulando 5 features)
    report_gen = DriftReportGeneratorEnhanced()
    p_values = {
        'feature_1': 0.001,
        'feature_2': 0.03,
        'feature_3': 0.08,
        'feature_4': 0.15,
        'feature_5': 0.45
    }
    
    correction_result = report_gen.apply_multiple_testing_correction(
        p_values, method='fdr_bh'
    )
    
    print("\n=== Correção de Múltiplos Testes ===")
    print(f"Significativos (raw): {correction_result['n_significant_raw']}")
    print(f"Significativos (corrigido): {correction_result['n_significant_corrected']}")
    print(f"Interpretação: {correction_result['interpretation']}")


if __name__ == "__main__":
    example_integration()
