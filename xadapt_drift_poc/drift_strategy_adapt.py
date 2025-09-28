# ESTRATÉGIAS DE ADAPTAÇÃO
# # ====================================

import time
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

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




"""
# XAdapt-Drift: Adaptação Inteligente e Semi-Automatizada

Esta seção implementa as estratégias de mitigação concretas e o sistema de decisão automatizado, completando os pilares fundamentais do framework XAdapt-Drift.

**Objetivos desta Implementação**

1. **Estratégias de Mitigação Concretas**: Transformar diagnósticos de drift em ações executáveis
2. **Sistema de Decisão Automatizado**: Selecionar estratégias baseadas em evidências científicas
3. **Ponte Diagnóstico → Ação**: Conectar a análise de drift com intervenções práticas

**Componentes Implementados**

- `AdaptationStrategy`: Classes base para estratégias de mitigação
- `IntelligentDecisionEngine`: Sistema de decisão baseado em regras e evidências
- `AdaptationOrchestrator`: Orquestrador que integra diagnóstico e adaptação
- Estratégias concretas: Retraining, Reweighting, Feature Engineering, Threshold Calibration

### METACLASSE E DATACLASSES 
- DriftSeverity(Enum)
- AdaptationAction(Enum)
- AdaptationContext (Dataclass)
- AdaptationResult (Dataclass)
- class AdaptationStrategy (Metaclasse):

### ESTRATÉGIAS DE ADAPTAÇÃO

- **ContinuousMonitoringStrategy** (AdaptationStrategy): Estratégia de monitoramento contínuo sem intervenção
- **ThresholdAdjustmentStrategy**  (AdaptationStrategy): Estratégia de ajuste de thresholds do modelo
- **FeatureRecalibrationStrategy** (AdaptationStrategy): Estratégia de recalibração de features específicas
- **PartialRetrainingStrategy**    (AdaptationStrategy): Estratégia de retreinamento parcial com dados recentes
- **FullRetrainingStrategy**       (AdaptationStrategy): Estratégia de retreinamento completo do modelo
- **EmergencyFallbackStrategy**    (AdaptationStrategy): Estratégia de fallback para situações críticas
"""


# ESTRATÉGIAS DE MITIGAÇÃO CONCRETAS
# =====================================

from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import warnings
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

class DriftSeverity(Enum):
    """Níveis de severidade do drift para tomada de decisão."""
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AdaptationAction(Enum):
    """Tipos de ações de adaptação disponíveis."""
    CONTINUE_MONITORING = "continue_monitoring"
    INCREASE_MONITORING = "increase_monitoring"
    FEATURE_RECALIBRATION = "feature_recalibration"
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"
    PARTIAL_RETRAINING = "partial_retraining"
    FULL_RETRAINING = "full_retraining"
    DATA_REWEIGHTING = "data_reweighting"
    FEATURE_SELECTION = "feature_selection"
    MODEL_REPLACEMENT = "model_replacement"
    EMERGENCY_FALLBACK = "emergency_fallback"

@dataclass
class AdaptationContext:
    """Contexto para tomada de decisão de adaptação."""
    drift_score: float
    drift_severity: DriftSeverity
    affected_features: List[str]
    performance_impact: float
    model_confidence: float
    business_criticality: str
    available_resources: Dict[str, Any]
    time_constraints: Optional[str] = None
    historical_drift_patterns: Optional[Dict] = None

@dataclass
class AdaptationResult:
    """Resultado da execução de uma estratégia de adaptação."""
    strategy_name: str
    action_taken: AdaptationAction
    success: bool
    performance_improvement: float
    execution_time: float
    metadata: Dict[str, Any]
    warnings: List[str]
    recommendations: List[str]

class AdaptationStrategy(ABC):
    """Classe base abstrata para estratégias de adaptação."""
    
    def __init__(self, name: str, priority: int = 5):
        self.name = name
        self.priority = priority  # 1 = highest priority, 10 = lowest
        self.execution_count = 0
        self.success_rate = 0.0
        self.average_improvement = 0.0
    
    @abstractmethod
    def can_apply(self, context: AdaptationContext) -> bool:
        """Verifica se esta estratégia pode ser aplicada no contexto atual."""
        pass
    
    @abstractmethod
    def estimate_impact(self, context: AdaptationContext) -> float:
        """Estima o impacto esperado desta estratégia (0-1)."""
        pass
    
    @abstractmethod
    def execute(self, model, X_train, y_train, X_current, 
                drift_metrics: Dict, context: AdaptationContext) -> AdaptationResult:
        """Executa a estratégia de adaptação."""
        pass
    
    @abstractmethod
    def get_cost_estimate(self, context: AdaptationContext) -> Dict[str, float]:
        """Estima custos de execução (computacional, tempo, recursos)."""
        pass
    
    def update_performance_metrics(self, success: bool, improvement: float):
        """Atualiza métricas de performance da estratégia."""
        self.execution_count += 1
        if success:
            self.success_rate = ((self.success_rate * (self.execution_count - 1)) + 1) / self.execution_count
            self.average_improvement = ((self.average_improvement * (self.execution_count - 1)) + improvement) / self.execution_count
        else:
            self.success_rate = (self.success_rate * (self.execution_count - 1)) / self.execution_count



class ContinuousMonitoringStrategy(AdaptationStrategy):
    """Estratégia de monitoramento contínuo sem intervenção."""
    
    def __init__(self):
        super().__init__("Continuous Monitoring", priority=10)  # Baixa prioridade
    
    def can_apply(self, context: AdaptationContext) -> bool:
        return context.drift_severity in [DriftSeverity.NEGLIGIBLE, DriftSeverity.LOW]
    
    def estimate_impact(self, context: AdaptationContext) -> float:
        return 0.1  # Impacto mínimo, apenas monitoramento
    
    def execute(self, model, X_train, y_train, X_current, 
                drift_metrics: Dict, context: AdaptationContext) -> AdaptationResult:
        start_time = time.time()
        
        # Apenas ajusta frequência de monitoramento
        monitoring_frequency = "daily" if context.drift_severity == DriftSeverity.LOW else "weekly"
        
        return AdaptationResult(
            strategy_name=self.name,
            action_taken=AdaptationAction.CONTINUE_MONITORING,
            success=True,
            performance_improvement=0.0,
            execution_time=time.time() - start_time,
            metadata={"monitoring_frequency": monitoring_frequency},
            warnings=[],
            recommendations=[f"Manter monitoramento {monitoring_frequency}"]
        )
    
    def get_cost_estimate(self, context: AdaptationContext) -> Dict[str, float]:
        return {"computational": 0.1, "time_hours": 0.0, "human_effort": 0.1}

class ThresholdAdjustmentStrategy(AdaptationStrategy):
    """Estratégia de ajuste de thresholds do modelo."""
    
    def __init__(self):
        super().__init__("Threshold Adjustment", priority=7)
    
    def can_apply(self, context: AdaptationContext) -> bool:
        return (context.drift_severity in [DriftSeverity.LOW, DriftSeverity.MEDIUM] and 
                hasattr(context.available_resources.get('model'), 'predict_proba'))
    
    def estimate_impact(self, context: AdaptationContext) -> float:
        return 0.3 if context.drift_severity == DriftSeverity.MEDIUM else 0.2
    
    def execute(self, model, X_train, y_train, X_current, 
                drift_metrics: Dict, context: AdaptationContext) -> AdaptationResult:
        start_time = time.time()
        
        try:
            # Simula ajuste de threshold baseado na severidade do drift
            if hasattr(model, 'predict_proba'):
                # Para drift baixo: ajuste conservador
                # Para drift médio: ajuste mais agressivo
                threshold_adjustment = 0.05 if context.drift_severity == DriftSeverity.LOW else 0.1
                
                return AdaptationResult(
                    strategy_name=self.name,
                    action_taken=AdaptationAction.THRESHOLD_ADJUSTMENT,
                    success=True,
                    performance_improvement=threshold_adjustment,
                    execution_time=time.time() - start_time,
                    metadata={"threshold_adjustment": threshold_adjustment},
                    warnings=[],
                    recommendations=["Monitorar performance pós-ajuste por 48h"]
                )
            else:
                return AdaptationResult(
                    strategy_name=self.name,
                    action_taken=AdaptationAction.THRESHOLD_ADJUSTMENT,
                    success=False,
                    performance_improvement=0.0,
                    execution_time=time.time() - start_time,
                    metadata={},
                    warnings=["Modelo não suporta predict_proba"],
                    recommendations=["Considerar estratégia alternativa"]
                )
                
        except Exception as e:
            return AdaptationResult(
                strategy_name=self.name,
                action_taken=AdaptationAction.THRESHOLD_ADJUSTMENT,
                success=False,
                performance_improvement=0.0,
                execution_time=time.time() - start_time,
                metadata={},
                warnings=[f"Erro na execução: {str(e)}"],
                recommendations=["Investigar causa do erro"]
            )
    
    def get_cost_estimate(self, context: AdaptationContext) -> Dict[str, float]:
        return {"computational": 0.3, "time_hours": 0.5, "human_effort": 0.2}

class FeatureRecalibrationStrategy(AdaptationStrategy):
    """Estratégia de recalibração de features específicas."""
    
    def __init__(self):
        super().__init__("Feature Recalibration", priority=5)
    
    def can_apply(self, context: AdaptationContext) -> bool:
        return (context.drift_severity in [DriftSeverity.MEDIUM, DriftSeverity.HIGH] and 
                len(context.affected_features) <= 5)  # Máximo 5 features afetadas
    
    def estimate_impact(self, context: AdaptationContext) -> float:
        impact_base = 0.4 if context.drift_severity == DriftSeverity.MEDIUM else 0.6
        feature_factor = min(1.0, len(context.affected_features) / 3)  # Mais features = maior impacto
        return impact_base * feature_factor
    
    def execute(self, model, X_train, y_train, X_current, 
                drift_metrics: Dict, context: AdaptationContext) -> AdaptationResult:
        start_time = time.time()
        
        try:
            # Simula recalibração baseada nas features afetadas
            recalibrated_features = []
            improvements = []
            
            for feature in context.affected_features:
                if feature in X_current.columns:
                    # Simula processo de recalibração
                    feature_improvement = np.random.uniform(0.05, 0.15)
                    recalibrated_features.append(feature)
                    improvements.append(feature_improvement)
            
            avg_improvement = np.mean(improvements) if improvements else 0.0
            
            return AdaptationResult(
                strategy_name=self.name,
                action_taken=AdaptationAction.FEATURE_RECALIBRATION,
                success=len(recalibrated_features) > 0,
                performance_improvement=avg_improvement,
                execution_time=time.time() - start_time,
                metadata={
                    "recalibrated_features": recalibrated_features,
                    "feature_improvements": dict(zip(recalibrated_features, improvements))
                },
                warnings=[],
                recommendations=[f"Validar recalibração de {len(recalibrated_features)} features"]
            )
            
        except Exception as e:
            return AdaptationResult(
                strategy_name=self.name,
                action_taken=AdaptationAction.FEATURE_RECALIBRATION,
                success=False,
                performance_improvement=0.0,
                execution_time=time.time() - start_time,
                metadata={},
                warnings=[f"Erro na recalibração: {str(e)}"],
                recommendations=["Verificar integridade dos dados"]
            )
    
    def get_cost_estimate(self, context: AdaptationContext) -> Dict[str, float]:
        feature_cost = len(context.affected_features) * 0.2
        return {"computational": feature_cost, "time_hours": 1.0, "human_effort": 0.5}


# ESTRATÉGIAS DE ALTO IMPACTO
# ==============================

class PartialRetrainingStrategy(AdaptationStrategy):
    """Estratégia de retreinamento parcial com dados recentes."""
    
    def __init__(self):
        super().__init__("Partial Retraining", priority=3)
    
    def can_apply(self, context: AdaptationContext) -> bool:
        return (context.drift_severity in [DriftSeverity.MEDIUM, DriftSeverity.HIGH] and
                context.available_resources.get('retraining_data_size', 0) > 100)
    
    def estimate_impact(self, context: AdaptationContext) -> float:
        base_impact = 0.6 if context.drift_severity == DriftSeverity.HIGH else 0.5
        data_factor = min(1.0, context.available_resources.get('retraining_data_size', 0) / 1000)
        return base_impact * (0.5 + 0.5 * data_factor)
    
    def execute(self, model, X_train, y_train, X_current, 
                drift_metrics: Dict, context: AdaptationContext) -> AdaptationResult:
        start_time = time.time()
        
        try:
            # Simula retreinamento parcial
            sample_size = min(len(X_current), context.available_resources.get('retraining_data_size', 500))
            
            if sample_size < 50:
                return AdaptationResult(
                    strategy_name=self.name,
                    action_taken=AdaptationAction.PARTIAL_RETRAINING,
                    success=False,
                    performance_improvement=0.0,
                    execution_time=time.time() - start_time,
                    metadata={},
                    warnings=["Dados insuficientes para retreinamento"],
                    recommendations=["Aguardar mais dados ou usar estratégia alternativa"]
                )
            
            # Simula processo de retreinamento
            training_time = sample_size * 0.001  # Simula tempo baseado no tamanho
            performance_gain = np.random.uniform(0.1, 0.3)
            
            return AdaptationResult(
                strategy_name=self.name,
                action_taken=AdaptationAction.PARTIAL_RETRAINING,
                success=True,
                performance_improvement=performance_gain,
                execution_time=time.time() - start_time + training_time,
                metadata={
                    "samples_used": sample_size,
                    "training_time": training_time,
                    "affected_features": context.affected_features
                },
                warnings=[],
                recommendations=["Validar modelo retreinado em holdout set"]
            )
            
        except Exception as e:
            return AdaptationResult(
                strategy_name=self.name,
                action_taken=AdaptationAction.PARTIAL_RETRAINING,
                success=False,
                performance_improvement=0.0,
                execution_time=time.time() - start_time,
                metadata={},
                warnings=[f"Erro no retreinamento: {str(e)}"],
                recommendations=["Verificar integridade do modelo e dados"]
            )
    
    def get_cost_estimate(self, context: AdaptationContext) -> Dict[str, float]:
        data_size = context.available_resources.get('retraining_data_size', 500)
        return {
            "computational": data_size * 0.002, 
            "time_hours": 2.0, 
            "human_effort": 1.0
        }

class FullRetrainingStrategy(AdaptationStrategy):
    """Estratégia de retreinamento completo do modelo."""
    
    def __init__(self):
        super().__init__("Full Retraining", priority=2)
    
    def can_apply(self, context: AdaptationContext) -> bool:
        return (context.drift_severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL] and
                context.available_resources.get('full_retraining_approved', False))
    
    def estimate_impact(self, context: AdaptationContext) -> float:
        return 0.8 if context.drift_severity == DriftSeverity.CRITICAL else 0.7
    
    def execute(self, model, X_train, y_train, X_current, 
                drift_metrics: Dict, context: AdaptationContext) -> AdaptationResult:
        start_time = time.time()
        
        try:
            # Simula retreinamento completo
            total_samples = len(X_train) + len(X_current)
            training_time = total_samples * 0.005  # Simula tempo extenso
            
            # Simula alta melhoria de performance
            performance_gain = np.random.uniform(0.3, 0.6)
            
            return AdaptationResult(
                strategy_name=self.name,
                action_taken=AdaptationAction.FULL_RETRAINING,
                success=True,
                performance_improvement=performance_gain,
                execution_time=time.time() - start_time + training_time,
                metadata={
                    "total_samples": total_samples,
                    "training_time": training_time,
                    "new_model_version": f"v{int(time.time())}"
                },
                warnings=["Modelo anterior será substituído"],
                recommendations=[
                    "Realizar testes A/B antes do deployment",
                    "Manter backup do modelo anterior",
                    "Monitorar performance intensivamente por 72h"
                ]
            )
            
        except Exception as e:
            return AdaptationResult(
                strategy_name=self.name,
                action_taken=AdaptationAction.FULL_RETRAINING,
                success=False,
                performance_improvement=0.0,
                execution_time=time.time() - start_time,
                metadata={},
                warnings=[f"Erro no retreinamento completo: {str(e)}"],
                recommendations=["Ativar fallback para modelo anterior"]
            )
    
    def get_cost_estimate(self, context: AdaptationContext) -> Dict[str, float]:
        return {"computational": 10.0, "time_hours": 8.0, "human_effort": 4.0}

class EmergencyFallbackStrategy(AdaptationStrategy):
    """Estratégia de fallback para situações críticas."""
    
    def __init__(self):
        super().__init__("Emergency Fallback", priority=1)  # Máxima prioridade
    
    def can_apply(self, context: AdaptationContext) -> bool:
        return context.drift_severity == DriftSeverity.CRITICAL
    
    def estimate_impact(self, context: AdaptationContext) -> float:
        return 0.9  # Alto impacto imediato
    
    def execute(self, model, X_train, y_train, X_current, 
                drift_metrics: Dict, context: AdaptationContext) -> AdaptationResult:
        start_time = time.time()
        
        try:
            # Ativa mecanismos de fallback
            fallback_actions = [
                "Ativar modelo de backup",
                "Reduzir complexidade de predições",
                "Aumentar conservadorismo nas decisões",
                "Alertar equipe de emergência"
            ]
            
            return AdaptationResult(
                strategy_name=self.name,
                action_taken=AdaptationAction.EMERGENCY_FALLBACK,
                success=True,
                performance_improvement=0.4,  # Estabilização, não otimização
                execution_time=time.time() - start_time,
                metadata={
                    "fallback_actions": fallback_actions,
                    "emergency_mode": True,
                    "alert_level": "CRITICAL"
                },
                warnings=["Sistema em modo de emergência"],
                recommendations=[
                    "Investigar causa raiz do drift crítico",
                    "Preparar plano de recuperação",
                    "Considerar intervenção manual"
                ]
            )
            
        except Exception as e:
            return AdaptationResult(
                strategy_name=self.name,
                action_taken=AdaptationAction.EMERGENCY_FALLBACK,
                success=False,
                performance_improvement=0.0,
                execution_time=time.time() - start_time,
                metadata={},
                warnings=[f"Falha no fallback: {str(e)}"],
                recommendations=["INTERVENÇÃO MANUAL NECESSÁRIA"]
            )
    
    def get_cost_estimate(self, context: AdaptationContext) -> Dict[str, float]:
        return {"computational": 0.5, "time_hours": 0.1, "human_effort": 2.0}




# 🤖 ADAPTATION STRATEGY ENGINE - BASEADO EM RELATÓRIO CIENTÍFICO
# ===============================================================
# Nova implementação que utiliza o relatório completo do DriftReportGenerator
# para tomar decisões de adaptação baseadas em evidências científicas

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class AdaptationStrategyEngine:
    """
    Engine de adaptação inteligente baseado no relatório científico do DriftReportGenerator.
    
    Esta versão utiliza diretamente as interpretações científicas e thresholds 
    já calculados pela DriftReportGenerator, eliminando duplicação de lógica
    e garantindo consistência com a metodologia científica estabelecida.
    """
    
    def __init__(self, auto_execution_threshold: float = 0.7):
        self.strategies = [
            EmergencyFallbackStrategy(),
            FullRetrainingStrategy(), 
            PartialRetrainingStrategy(),
            FeatureRecalibrationStrategy(),
            ThresholdAdjustmentStrategy(),
            ContinuousMonitoringStrategy()
        ]
        
        # Ordenar por prioridade
        self.strategies.sort(key=lambda x: x.priority)
        
        self.auto_execution_threshold = auto_execution_threshold
        self.execution_history = []
        
        # Mapeamento de severidade do relatório para DriftSeverity enum
        self.severity_mapping = {
            'NEGLIGIBLE': DriftSeverity.NEGLIGIBLE,
            'LOW': DriftSeverity.LOW,
            'MEDIUM': DriftSeverity.MEDIUM,
            'HIGH': DriftSeverity.HIGH,
            'CRITICAL': DriftSeverity.CRITICAL
        }
    
    def extract_drift_context_from_report(self, drift_report: Dict, business_context: Dict) -> AdaptationContext:
        """
        Extrai contexto de adaptação diretamente do relatório científico.
        
        Args:
            drift_report: Relatório completo do DriftReportGenerator
            business_context: Contexto de negócio adicional
            
        Returns:
            AdaptationContext: Contexto estruturado para tomada de decisão
        """
        
        # 1. ANÁLISE DO EXECUTIVE SUMMARY
        executive_summary = drift_report.get('executive_summary', {})
        
        # Calcular score geral baseado nos achados de alta confiança
        total_features = executive_summary.get('total_features_analyzed', 1)
        high_confidence_findings = executive_summary.get('high_confidence_findings', 0)
        features_with_drift = executive_summary.get('features_with_significant_drift', 0)
        
        # Score baseado na proporção de features afetadas e confiança dos achados
        drift_score = (features_with_drift / total_features) * 0.6 + \
                     (high_confidence_findings / total_features) * 0.4
        
        # 2. DETERMINAR SEVERIDADE GERAL
        primary_concerns = executive_summary.get('primary_concerns', [])
        
        if not primary_concerns:
            overall_severity = DriftSeverity.NEGLIGIBLE
        else:
            # Usar a maior severidade encontrada nas preocupações primárias
            severities = [concern.get('severity', 'LOW') for concern in primary_concerns]
            max_severity_str = max(severities, key=lambda x: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].index(x))
            overall_severity = self.severity_mapping.get(max_severity_str, DriftSeverity.MEDIUM)
        
        # 3. IDENTIFICAR FEATURES MAIS AFETADAS
        affected_features = []
        feature_priorities = {}
        
        for concern in primary_concerns:
            feature = concern.get('feature')
            severity = concern.get('severity', 'LOW')
            
            if feature and feature not in affected_features:
                affected_features.append(feature)
                
            # Priorizar features por severidade
            severity_weight = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
            current_weight = feature_priorities.get(feature, 0)
            feature_priorities[feature] = max(current_weight, severity_weight.get(severity, 1))
        
        # Ordenar features por prioridade e limitar a 5
        sorted_features = sorted(feature_priorities.items(), key=lambda x: x[1], reverse=True)
        affected_features = [feature for feature, _ in sorted_features[:5]]
        
        # 4. CALCULAR IMPACTO DE PERFORMANCE ESTIMADO
        # Baseado na quantidade e severidade dos achados
        performance_impact = self._estimate_performance_impact(
            drift_score, overall_severity, len(affected_features)
        )
        
        # 5. CRIAR CONTEXTO DE ADAPTAÇÃO
        return AdaptationContext(
            drift_score=drift_score,
            drift_severity=overall_severity,
            affected_features=affected_features,
            performance_impact=performance_impact,
            model_confidence=self._estimate_model_confidence(drift_report),
            business_criticality=business_context.get('criticality', 'medium'),
            available_resources={
                'retraining_data_size': business_context.get('data_size', 1000),
                'full_retraining_approved': business_context.get('auto_retrain', False),
                'model': business_context.get('model')
            },
            time_constraints=business_context.get('time_limit'),
            historical_drift_patterns=self._get_historical_patterns()
        )
    
    def _estimate_performance_impact(self, drift_score: float, severity: DriftSeverity, 
                                   num_affected_features: int) -> float:
        """Estima impacto de performance baseado nas métricas do relatório."""
        
        base_impact = {
            DriftSeverity.NEGLIGIBLE: 0.01,
            DriftSeverity.LOW: 0.05,
            DriftSeverity.MEDIUM: 0.15,
            DriftSeverity.HIGH: 0.30,
            DriftSeverity.CRITICAL: 0.50
        }.get(severity, 0.10)
        
        # Ajustar baseado no número de features afetadas
        feature_multiplier = min(1.0 + (num_affected_features - 1) * 0.1, 2.0)
        
        return min(base_impact * feature_multiplier, 0.80)  # Máximo 80% de impacto
    
    def _estimate_model_confidence(self, drift_report: Dict) -> float:
        """Estima confiança do modelo baseada no relatório de drift."""
        
        # Analisar confiança das interpretações científicas
        drift_analysis = drift_report.get('drift_analysis', {})
        confidence_scores = []
        
        for feature, analysis in drift_analysis.items():
            metric_interpretations = analysis.get('metric_interpretations', {})
            
            for metric, interpretation in metric_interpretations.items():
                confidence = interpretation.get('confidence', 'MODERATE')
                
                confidence_value = {
                    'VERY_HIGH': 0.95,
                    'HIGH': 0.85,
                    'MODERATE': 0.70,
                    'LOW': 0.50
                }.get(confidence, 0.70)
                
                confidence_scores.append(confidence_value)
        
        if confidence_scores:
            avg_confidence = np.mean(confidence_scores)
            # Inverter: alta confiança nas métricas = baixa confiança no modelo atual
            return max(0.1, 1.0 - avg_confidence * 0.5)
        
        return 0.80  # Confiança padrão
    
    def _get_historical_patterns(self) -> Dict:
        """Analisa padrões históricos."""
        if not self.execution_history:
            return {}
        
        recent_executions = [h for h in self.execution_history 
                           if h['timestamp'] > datetime.now() - timedelta(days=30)]
        
        return {
            'recent_adaptations': len(recent_executions),
            'success_rate': np.mean([h['success'] for h in recent_executions]) if recent_executions else 0.0,
            'avg_improvement': np.mean([h['improvement'] for h in recent_executions]) if recent_executions else 0.0
        }

    def extract_feature_level_insights(self, drift_report: Dict) -> Dict[str, Dict]:
        """
        Extrai insights específicos por feature do relatório científico.
        
        Args:
            drift_report: Relatório completo do DriftReportGenerator
            
        Returns:
            Dict com insights detalhados por feature
        """
        feature_insights = {}
        
        drift_analysis = drift_report.get('drift_analysis', {})
        
        for feature, analysis in drift_analysis.items():
            insights = {
                'drift_detected': False,
                'severity': 'NEGLIGIBLE',
                'confidence': 'LOW',
                'affected_metrics': [],
                'adaptation_priority': 'low',
                'specific_recommendations': []
            }
            
            # Analisar interpretações das métricas
            metric_interpretations = analysis.get('metric_interpretations', {})
            
            for metric_name, interpretation in metric_interpretations.items():
                status = interpretation.get('status', 'NO_DRIFT')
                
                if status in ['SIGNIFICANT_DRIFT', 'CRITICAL_DRIFT']:
                    insights['drift_detected'] = True
                    insights['affected_metrics'].append(metric_name)
                    
                    # Atualizar severidade se maior
                    current_severity = interpretation.get('severity', 'LOW')
                    if self._compare_severity(current_severity, insights['severity']) > 0:
                        insights['severity'] = current_severity
                    
                    # Atualizar confiança
                    confidence = interpretation.get('confidence', 'LOW')
                    if self._compare_confidence(confidence, insights['confidence']) > 0:
                        insights['confidence'] = confidence
                    
                    # Extrair recomendações específicas
                    recommendation = interpretation.get('recommendation', '')
                    if recommendation and recommendation not in insights['specific_recommendations']:
                        insights['specific_recommendations'].append(recommendation)
            
            # Determinar prioridade de adaptação
            if insights['drift_detected']:
                severity_priority = {
                    'LOW': 'low',
                    'MEDIUM': 'medium', 
                    'HIGH': 'high',
                    'CRITICAL': 'urgent'
                }
                insights['adaptation_priority'] = severity_priority.get(insights['severity'], 'low')
            
            feature_insights[feature] = insights
        
        return feature_insights
    
    def _compare_severity(self, severity1: str, severity2: str) -> int:
        """Compara duas severidades. Retorna 1 se severity1 > severity2, -1 se menor, 0 se igual."""
        order = ['NEGLIGIBLE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        idx1 = order.index(severity1) if severity1 in order else 0
        idx2 = order.index(severity2) if severity2 in order else 0
        
        if idx1 > idx2:
            return 1
        elif idx1 < idx2:
            return -1
        else:
            return 0
    
    def _compare_confidence(self, confidence1: str, confidence2: str) -> int:
        """Compara duas confianças."""
        order = ['LOW', 'MODERATE', 'HIGH', 'VERY_HIGH']
        idx1 = order.index(confidence1) if confidence1 in order else 0
        idx2 = order.index(confidence2) if confidence2 in order else 0
        
        if idx1 > idx2:
            return 1
        elif idx1 < idx2:
            return -1
        else:
            return 0
    
    def select_optimal_strategy_from_report(self, drift_report: Dict, 
                                           business_context: Dict) -> Tuple[AdaptationStrategy, float]:
        """
        Seleciona estratégia ótima baseada no relatório científico completo.
        
        Args:
            drift_report: Relatório do DriftReportGenerator
            business_context: Contexto de negócio
            
        Returns:
            Tuple com (estratégia_selecionada, score_confiança)
        """
        
        # 1. Extrair contexto do relatório
        context = self.extract_drift_context_from_report(drift_report, business_context)
        
        # 2. Extrair insights por feature
        feature_insights = self.extract_feature_level_insights(drift_report)
        
        # 3. Avaliar cada estratégia baseada nas evidências científicas
        strategy_scores = {}
        
        for strategy in self.strategies:
            # Verificar se a estratégia pode ser aplicada
            if not strategy.can_apply(context):
                strategy_scores[strategy] = 0.0
                print(f"📊 {strategy.__class__.__name__}:")
                print(f"   ❌ Não aplicável para o contexto atual")
                print(f"   🎯 Score Final: 0.000\n")
                continue
            
            # Score baseado no impacto estimado da estratégia
            impact_score = strategy.estimate_impact(context)
            
            # Ajustar score baseado nos insights específicos das features
            feature_adjustment = self._calculate_feature_adjustment(strategy, feature_insights)
            
            # Score de viabilidade baseado no relatório
            viability_score = self._calculate_viability_from_report(strategy, drift_report, business_context)
            
            # Calcular score de custo (inverso do custo total)
            cost_estimate = strategy.get_cost_estimate(context)
            total_cost = sum(cost_estimate.values())
            cost_score = max(0.1, 1.0 - (total_cost / 20.0))  # Normalizar custo
            
            # Score final ponderado
            final_score = (impact_score * 0.4 + 
                          feature_adjustment * 0.3 + 
                          viability_score * 0.2 +
                          cost_score * 0.1)
            
            strategy_scores[strategy] = final_score
            
            print(f"📊 {strategy.__class__.__name__}:")
            print(f"   Impacto: {impact_score:.3f}")
            print(f"   Ajuste Features: {feature_adjustment:.3f}")
            print(f"   Viabilidade: {viability_score:.3f}")
            print(f"   Custo: {cost_score:.3f}")
            print(f"   🎯 Score Final: {final_score:.3f}\n")
        
        # 4. Selecionar melhor estratégia
        best_strategy = max(strategy_scores.keys(), key=lambda s: strategy_scores[s])
        confidence_score = strategy_scores[best_strategy]
        
        return best_strategy, confidence_score
    
    def _calculate_feature_adjustment(self, strategy: AdaptationStrategy, 
                                     feature_insights: Dict[str, Dict]) -> float:
        """
        Calcula ajuste baseado nos insights específicos das features.
        """
        if not feature_insights:
            return 0.5  # Score neutro
        
        total_adjustment = 0.0
        feature_count = len(feature_insights)
        
        for feature, insights in feature_insights.items():
            if not insights['drift_detected']:
                continue
                
            # Ajuste baseado na severidade da feature
            severity_weight = {
                'LOW': 0.2,
                'MEDIUM': 0.5,
                'HIGH': 0.8,
                'CRITICAL': 1.0
            }.get(insights['severity'], 0.3)
            
            # Ajuste baseado na adequação da estratégia para esta feature
            strategy_feature_fit = self._evaluate_strategy_for_feature(strategy, insights)
            
            total_adjustment += severity_weight * strategy_feature_fit
        
        return min(total_adjustment / feature_count if feature_count > 0 else 0.5, 1.0)
    
    def _evaluate_strategy_for_feature(self, strategy: AdaptationStrategy, 
                                      feature_insight: Dict) -> float:
        """Avalia adequação da estratégia para uma feature específica."""
        
        strategy_name = strategy.__class__.__name__
        severity = feature_insight['severity']
        affected_metrics = feature_insight['affected_metrics']
        
        # Regras específicas baseadas no tipo de estratégia e problema detectado
        if strategy_name == 'EmergencyFallbackStrategy':
            return 1.0 if severity == 'CRITICAL' else 0.2
            
        elif strategy_name == 'FullRetrainingStrategy':
            # Mais adequado para múltiplas métricas afetadas ou alta severidade
            if len(affected_metrics) >= 3 or severity in ['HIGH', 'CRITICAL']:
                return 0.9
            return 0.4
            
        elif strategy_name == 'PartialRetrainingStrategy':
            # Adequado para algumas features afetadas
            if 1 <= len(affected_metrics) <= 2 and severity in ['MEDIUM', 'HIGH']:
                return 0.8
            return 0.5
            
        elif strategy_name == 'FeatureRecalibrationStrategy':
            # Específico para problemas de features individuais
            if len(affected_metrics) <= 2 and severity in ['LOW', 'MEDIUM']:
                return 0.7
            return 0.3
            
        elif strategy_name == 'ThresholdAdjustmentStrategy':
            # Para drift leve que pode ser corrigido com ajustes
            if severity in ['LOW', 'MEDIUM']:
                return 0.6
            return 0.2
            
        elif strategy_name == 'ContinuousMonitoringStrategy':
            # Para casos onde não há drift significativo
            if severity in ['NEGLIGIBLE', 'LOW']:
                return 0.8
            return 0.1
        
        return 0.5  # Score padrão
    
    def _calculate_viability_from_report(self, strategy: AdaptationStrategy, 
                                        drift_report: Dict, business_context: Dict) -> float:
        """
        Calcula viabilidade da estratégia baseada no relatório e contexto de negócio.
        """
        
        # Fatores de viabilidade
        viability_factors = []
        
        # 1. Disponibilidade de dados (baseado no executive summary)
        executive_summary = drift_report.get('executive_summary', {})
        total_features = executive_summary.get('total_features_analyzed', 0)
        
        if total_features > 0:
            data_availability = min(total_features / 10.0, 1.0)  # Normalizar para max 10 features
            viability_factors.append(data_availability)
        
        # 2. Recursos disponíveis
        available_resources = business_context.get('available_resources', {})
        
        if strategy.__class__.__name__ in ['FullRetrainingStrategy', 'PartialRetrainingStrategy']:
            # Verificar se modelo está disponível
            model_available = 1.0 if available_resources.get('model') is not None else 0.3
            viability_factors.append(model_available)
            
            # Verificar aprovação para retreinamento
            retrain_approved = 1.0 if available_resources.get('full_retraining_approved', False) else 0.5
            viability_factors.append(retrain_approved)
        
        # 3. Restrições de tempo
        time_constraints = business_context.get('time_constraints')
        if time_constraints:
            if strategy.__class__.__name__ == 'EmergencyFallbackStrategy':
                time_viability = 1.0  # Sempre viável para emergência
            elif strategy.__class__.__name__ in ['FullRetrainingStrategy']:
                time_viability = 0.3 if time_constraints == 'urgent' else 0.8
            else:
                time_viability = 0.8
            viability_factors.append(time_viability)
        
        # 4. Criticidade do negócio
        criticality = business_context.get('criticality', 'medium')
        if criticality == 'high' and strategy.__class__.__name__ != 'EmergencyFallbackStrategy':
            viability_factors.append(0.7)  # Reduzir viabilidade de estratégias arriscadas
        else:
            viability_factors.append(1.0)
        
        return np.mean(viability_factors) if viability_factors else 0.5

    def execute_adaptation_from_report(self, drift_report: Dict, business_context: Dict,
                                      auto_execute: bool = False) -> Dict:
        """
        Executa adaptação baseada no relatório científico completo.
        
        Args:
            drift_report: Relatório completo do DriftReportGenerator
            business_context: Contexto de negócio com modelo, dados, etc.
            auto_execute: Se True, executa automaticamente se confiança > threshold
            
        Returns:
            Dict com resultado da execução e detalhes
        """
        
        start_time = datetime.now()
        
        print("🔬 EXECUTANDO ADAPTAÇÃO BASEADA EM RELATÓRIO CIENTÍFICO")
        print("=" * 60)
        
        try:
            # 1. ANÁLISE DO EXECUTIVE SUMMARY
            executive_summary = drift_report.get('executive_summary', {})
            print(f"📋 Executive Summary:")
            print(f"   Features analisadas: {executive_summary.get('total_features_analyzed', 0)}")
            print(f"   Features com drift: {executive_summary.get('features_with_significant_drift', 0)}")
            print(f"   Achados de alta confiança: {executive_summary.get('high_confidence_findings', 0)}")
            
            # 2. PREOCUPAÇÕES PRIMÁRIAS
            primary_concerns = executive_summary.get('primary_concerns', [])
            if primary_concerns:
                print(f"\n🚨 Preocupações Primárias ({len(primary_concerns)}):")
                for i, concern in enumerate(primary_concerns[:3], 1):  # Mostrar apenas as 3 principais
                    print(f"   {i}. {concern.get('feature', 'Unknown')} - {concern.get('severity', 'LOW')}")
                    print(f"      Problema: {concern.get('issue', 'Drift detectado')}")
            
            # 3. SELEÇÃO DE ESTRATÉGIA
            print(f"\n🎯 SELECIONANDO ESTRATÉGIA ÓTIMA:")
            print("-" * 40)
            
            best_strategy, confidence = self.select_optimal_strategy_from_report(
                drift_report, business_context
            )
            
            print(f"✅ Estratégia Selecionada: {best_strategy.__class__.__name__}")
            print(f"🎯 Confiança: {confidence:.3f}")
            
            # 4. DECISÃO DE EXECUÇÃO
            execution_decision = self._make_execution_decision(
                best_strategy, confidence, auto_execute, drift_report
            )
            
            print(f"\n📋 Decisão de Execução: {execution_decision['decision']}")
            print(f"💡 Justificativa: {execution_decision['justification']}")
            
            # 5. EXECUÇÃO (SE APROVADA)
            execution_result = None
            if execution_decision['execute']:
                print(f"\n🚀 EXECUTANDO ESTRATÉGIA: {best_strategy.__class__.__name__}")
                print("-" * 50)
                
                # Extrair contexto para execução
                context = self.extract_drift_context_from_report(drift_report, business_context)
                
                try:
                    # Usar parâmetros corretos para execute() das estratégias existentes
                    execution_result = best_strategy.execute(
                        model=business_context.get('model'),
                        X_train=None,  # Simular dados
                        y_train=None,  # Simular dados
                        X_current=None,  # Simular dados
                        drift_metrics={},  # Simular métricas
                        context=context
                    )
                    
                    print(f"✅ Execução concluída com sucesso!")
                    print(f"📈 Melhoria estimada: {execution_result.performance_improvement:.3f}")
                    
                except Exception as e:
                    print(f"❌ Erro na execução: {str(e)}")
                    execution_result = {
                        'success': False,
                        'error': str(e),
                        'performance_improvement': 0.0
                    }
            
            # 6. REGISTRAR HISTÓRICO
            execution_record = {
                'timestamp': start_time,
                'strategy': best_strategy.__class__.__name__,
                'confidence': confidence,
                'executed': execution_decision['execute'],
                'success': getattr(execution_result, 'success', False) if execution_result else False,
                'improvement': getattr(execution_result, 'performance_improvement', 0.0) if execution_result else 0.0,
                'drift_report_summary': {
                    'total_features': executive_summary.get('total_features_analyzed', 0),
                    'features_with_drift': executive_summary.get('features_with_significant_drift', 0),
                    'primary_concerns_count': len(primary_concerns)
                }
            }
            
            self.execution_history.append(execution_record)
            
            # 7. RESULTADO FINAL
            duration = datetime.now() - start_time
            
            result = {
                'strategy_selected': best_strategy.__class__.__name__,
                'confidence_score': confidence,
                'execution_decision': execution_decision,
                'execution_result': execution_result,
                'execution_time': duration.total_seconds(),
                'drift_summary': {
                    'total_features_analyzed': executive_summary.get('total_features_analyzed', 0),
                    'features_with_drift': executive_summary.get('features_with_significant_drift', 0),
                    'high_confidence_findings': executive_summary.get('high_confidence_findings', 0),
                    'primary_concerns': len(primary_concerns)
                },
                'feature_insights': self.extract_feature_level_insights(drift_report)
            }
            
            print(f"\n🏁 ADAPTAÇÃO CONCLUÍDA")
            print(f"⏱️  Tempo total: {duration.total_seconds():.2f}s")
            print("=" * 60)
            
            return result
            
        except Exception as e:
            error_msg = f"Erro crítico na execução da adaptação: {str(e)}"
            print(f"💥 {error_msg}")
            
            return {
                'error': error_msg,
                'strategy_selected': None,
                'confidence_score': 0.0,
                'execution_decision': {'execute': False, 'decision': 'ERROR'},
                'execution_result': None,
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _make_execution_decision(self, strategy: AdaptationStrategy, confidence: float,
                               auto_execute: bool, drift_report: Dict) -> Dict:
        """
        Toma decisão sobre executar ou não a estratégia baseada no relatório.
        """
        
        # Fatores para decisão
        executive_summary = drift_report.get('executive_summary', {})
        primary_concerns = executive_summary.get('primary_concerns', [])
        
        # 1. Verificar se há drift crítico
        has_critical_drift = any(
            concern.get('severity') == 'CRITICAL' 
            for concern in primary_concerns
        )
        
        # 2. Verificar confiança mínima
        confidence_acceptable = confidence >= self.auto_execution_threshold
        
        # 3. Estratégia de emergência sempre executa se há drift crítico
        is_emergency_strategy = strategy.__class__.__name__ == 'EmergencyFallbackStrategy'
        
        # 4. Decisão
        if has_critical_drift and is_emergency_strategy:
            return {
                'execute': True,
                'decision': 'AUTO_EXECUTE_CRITICAL',
                'justification': 'Drift crítico detectado, executando estratégia de emergência automaticamente'
            }
        
        elif auto_execute and confidence_acceptable:
            return {
                'execute': True,
                'decision': 'AUTO_EXECUTE_CONFIDENCE',
                'justification': f'Execução automática aprovada (confiança: {confidence:.3f})'
            }
        
        elif confidence >= 0.8:  # Alta confiança
            return {
                'execute': not auto_execute,  # Só executa se não for auto (precisa confirmação manual)
                'decision': 'RECOMMEND_EXECUTE',
                'justification': f'Alta confiança na estratégia (confiança: {confidence:.3f}). Recomendada execução manual.'
            }
        
        elif confidence >= 0.5:  # Confiança moderada
            return {
                'execute': False,
                'decision': 'REQUIRE_REVIEW',
                'justification': f'Confiança moderada (confiança: {confidence:.3f}). Requer revisão humana antes da execução.'
            }
        
        else:  # Baixa confiança
            return {
                'execute': False,
                'decision': 'NOT_RECOMMENDED',
                'justification': f'Baixa confiança na estratégia (confiança: {confidence:.3f}). Execução não recomendada.'
            }
    
    def get_adaptation_recommendations(self, drift_report: Dict) -> List[str]:
        """
        Gera recomendações de adaptação baseadas no relatório científico.
        """
        recommendations = []
        
        # Analisar executive summary
        executive_summary = drift_report.get('executive_summary', {})
        primary_concerns = executive_summary.get('primary_concerns', [])
        
        if not primary_concerns:
            recommendations.append("✅ Nenhum drift significativo detectado. Manter monitoramento contínuo.")
            return recommendations
        
        # Recomendações baseadas nas preocupações primárias
        for concern in primary_concerns:
            feature = concern.get('feature', 'Feature desconhecida')
            severity = concern.get('severity', 'LOW')
            issue = concern.get('issue', 'Drift detectado')
            
            if severity == 'CRITICAL':
                recommendations.append(f"🚨 CRÍTICO - {feature}: {issue}. Ação imediata necessária.")
            elif severity == 'HIGH':
                recommendations.append(f"⚠️ ALTO - {feature}: {issue}. Priorizar correção.")
            elif severity == 'MEDIUM':
                recommendations.append(f"📊 MÉDIO - {feature}: {issue}. Monitorar e planejar correção.")
            else:
                recommendations.append(f"ℹ️ BAIXO - {feature}: {issue}. Incluir em próxima manutenção.")
        
        # Recomendação geral baseada no número de features afetadas
        features_with_drift = executive_summary.get('features_with_significant_drift', 0)
        total_features = executive_summary.get('total_features_analyzed', 1)
        
        drift_ratio = features_with_drift / total_features
        
        if drift_ratio > 0.5:
            recommendations.append("🔄 Recomendado: Retreinamento completo do modelo devido ao drift generalizado.")
        elif drift_ratio > 0.2:
            recommendations.append("🎯 Recomendado: Retreinamento parcial focado nas features afetadas.")
        else:
            recommendations.append("🔧 Recomendado: Recalibração específica das features com drift.")
        
        return recommendations