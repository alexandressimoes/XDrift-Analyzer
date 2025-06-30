import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import os

class ReportGenerator:
    """Generates reports from drift analysis results."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def generate(self, 
                drift_results: Dict[str, Dict[str, Any]],
                characterization: Optional[Dict[str, Dict[str, Any]]] = None,
                impact_analysis: Optional[Dict[str, Any]] = None,
                report_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive report.
        
        Args:
            drift_results: Results from DriftDetector
            characterization: Results from DriftCharacterizer
            impact_analysis: Results from ImpactAnalyzer
            report_name: Name for report file
            
        Returns:
            Report as dictionary
        """
        # Prepare report structure
        timestamp = datetime.now().isoformat()
        
        report = {
            "metadata": {
                "timestamp": timestamp,
                "version": "1.0"
            },
            "drift_summary": self._generate_drift_summary(drift_results),
            "drift_details": drift_results
        }
        
        # Add characterization if available
        if characterization:
            report["characterization"] = characterization
            
        # Add impact analysis if available
        if impact_analysis:
            report["impact_analysis"] = impact_analysis
            
        # Generate executive summary
        report["executive_summary"] = self._generate_executive_summary(
            drift_results, characterization, impact_analysis
        )
        
        # Save report if output directory specified
        if self.output_dir:
            if not report_name:
                report_name = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
            file_path = os.path.join(self.output_dir, report_name)
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2, default=self._json_serializable)
                
        return report
    
    def _json_serializable(self, obj):
        """Make objects JSON serializable."""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')
    
    def _generate_drift_summary(self, drift_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of drift detection results."""
        # Count drifted features
        drifted_features = [f for f, r in drift_results.items() if r["drift_detected"]]
        n_drifted = len(drifted_features)
        n_total = len(drift_results)
        
        # Categorize by feature type
        numerical_drifted = [f for f in drifted_features 
                            if drift_results[f]["feature_type"] == "numerical"]
        categorical_drifted = [f for f in drifted_features 
                              if drift_results[f]["feature_type"] == "categorical"]
        
        return {
            "total_features_analyzed": n_total,
            "drifted_features_count": n_drifted,
            "drift_ratio": n_drifted / n_total if n_total > 0 else 0,
            "numerical_drifted_count": len(numerical_drifted),
            "categorical_drifted_count": len(categorical_drifted),
            "drifted_features": drifted_features
        }
        
    def _generate_executive_summary(self, 
                                   drift_results: Dict[str, Dict[str, Any]],
                                   characterization: Optional[Dict[str, Dict[str, Any]]],
                                   impact_analysis: Optional[Dict[str, Any]]) -> str:
        """Generate an executive summary combining all analyses."""
        summary_parts = []
        
        # Summarize drift detection
        drifted_features = [f for f, r in drift_results.items() if r["drift_detected"]]
        n_drifted = len(drifted_features)
        n_total = len(drift_results)
        
        if n_drifted == 0:
            summary_parts.append(f"No drift detected across {n_total} analyzed features.")
            return " ".join(summary_parts)
            
        summary_parts.append(f"Drift detected in {n_drifted} out of {n_total} features.")
        
        # Add impact summary if available
        if impact_analysis and "impact_summary" in impact_analysis:
            summary_parts.append(impact_analysis["impact_summary"])
            
        # Add characterization insights if available
        if characterization and n_drifted > 0:
            # Add information about most significant drifted feature
            if impact_analysis and "high_impact_features" in impact_analysis and impact_analysis["high_impact_features"]:
                high_impact = impact_analysis["high_impact_features"][0]
                if high_impact in characterization:
                    summary_parts.append(f"Most impactful drift in '{high_impact}': " + 
                                       characterization[high_impact]["summary"])
            elif n_drifted > 0:
                # Just take the first drifted feature
                first_feature = drifted_features[0]
                if first_feature in characterization:
                    summary_parts.append(f"Example drift in '{first_feature}': " + 
                                       characterization[first_feature]["summary"])
        
        return " ".join(summary_parts)
