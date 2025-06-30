from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd

from xadapt_drift.adapters.base import BaseAdapter
from xadapt_drift.drift.detector import DriftDetector
from xadapt_drift.drift.characterizer import DriftCharacterizer
from xadapt_drift.impact.analyzer import ImpactAnalyzer
from xadapt_drift.report.generator import ReportGenerator

__version__ = "0.1.0"

class XAdaptDrift:
    """Main entry point for the XAdapt-Drift framework.
    
    This class provides an integrated interface for drift detection,
    characterization, and impact analysis.
    """
    
    def __init__(self, model_adapter: Optional[BaseAdapter] = None, 
                 drift_threshold: float = 0.05,
                 explanation_method: str = "shap",
                 output_dir: Optional[str] = None):
        """
        Args:
            model_adapter: Adapter for the ML model
            drift_threshold: p-value threshold for drift detection
            explanation_method: Method for feature importance ("shap" or "permutation")
            output_dir: Directory to save reports
        """
        self.model_adapter = model_adapter
        self.drift_detector = DriftDetector(threshold=drift_threshold)
        self.drift_characterizer = DriftCharacterizer()
        self.impact_analyzer = ImpactAnalyzer()
        self.report_generator = ReportGenerator(output_dir=output_dir)
        self.explanation_method = explanation_method
        
    def analyze(self, 
               reference: pd.DataFrame, 
               current: pd.DataFrame,
               y_reference: Optional[np.ndarray] = None,
               y_current: Optional[np.ndarray] = None,
               features: Optional[List[str]] = None,
               report_name: Optional[str] = None) -> Dict[str, Any]:
        """Run complete drift analysis pipeline.
        
        Args:
            reference: Reference data
            current: Current data to analyze for drift
            y_reference: Reference labels (for permutation importance)
            y_current: Current labels (for permutation importance)
            features: Features to analyze (if None, analyzes all common features)
            report_name: Name for the output report file
            
        Returns:
            Complete analysis report
        """
        # 1. Drift Detection
        drift_results = self.drift_detector.detect(reference, current, features)
        
        # 2. Drift Characterization
        characterization = self.drift_characterizer.characterize(
            drift_results, reference, current
        )
        
        # 3. Impact Analysis (if model adapter is provided)
        impact_analysis = None
        if self.model_adapter:
            impact_analysis = self.impact_analyzer.analyze(
                drift_results, reference, current, self.model_adapter,
                y_reference, y_current, self.explanation_method
            )
        
        # 4. Generate Report
        report = self.report_generator.generate(
            drift_results, characterization, impact_analysis, report_name
        )
        
        return report
    
    def detect_drift(self, reference: pd.DataFrame, current: pd.DataFrame, 
                   features: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Only run drift detection step."""
        return self.drift_detector.detect(reference, current, features)
        
    def characterize_drift(self, drift_results: Dict[str, Dict[str, Any]],
                          reference: pd.DataFrame, 
                          current: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Only run drift characterization step."""
        return self.drift_characterizer.characterize(drift_results, reference, current)
        
    def analyze_impact(self, drift_results: Dict[str, Dict[str, Any]],
                       reference: pd.DataFrame, 
                       current: pd.DataFrame,
                       y_reference: Optional[np.ndarray] = None,
                       y_current: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Only run impact analysis step."""
        if self.model_adapter is None:
            raise ValueError("Model adapter is required for impact analysis")
            
        return self.impact_analyzer.analyze(
            drift_results, reference, current, self.model_adapter,
            y_reference, y_current, self.explanation_method
        )