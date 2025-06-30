import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union

from xadapt_drift.adapters.base import BaseAdapter

class ImpactAnalyzer:
    """Analyzes the impact of drift on model behavior using feature importance."""
    
    def analyze(self, 
                drift_results: Dict[str, Dict[str, Any]],
                reference: pd.DataFrame, 
                current: pd.DataFrame,
                model_adapter: BaseAdapter,
                y_reference: Optional[np.ndarray] = None,
                y_current: Optional[np.ndarray] = None,
                explanation_method: str = "shap") -> Dict[str, Any]:
        """Analyze impact of drift on model behavior.
        
        Args:
            drift_results: Results from DriftDetector
            reference: Reference data
            current: Current data 
            model_adapter: Model adapter for explanations
            y_reference: Reference target values (for permutation importance)
            y_current: Current target values (for permutation importance)
            explanation_method: Method to use for explanations ("shap" or "permutation")
            
        Returns:
            Dictionary with impact analysis results
        """
        # Get feature importances for reference data
        if explanation_method == "permutation" and y_reference is None:
            raise ValueError("y_reference must be provided for permutation importance")
            
        reference_importances = model_adapter.explain(
            reference.values,
            y=y_reference if explanation_method == "permutation" else None,
            method=explanation_method
        )
        
        # Get feature importances for current data
        if explanation_method == "permutation" and y_current is None:
            raise ValueError("y_current must be provided for permutation importance")
            
        current_importances = model_adapter.explain(
            current.values, 
            y=y_current if explanation_method == "permutation" else None,
            method=explanation_method
        )
        
        # Calculate Drift Impact Score (DIS) for each feature
        feature_impact = {}
        for feature in model_adapter.feature_names:
            if feature not in reference_importances or feature not in current_importances:
                continue
                
            ref_imp = reference_importances[feature]
            curr_imp = current_importances[feature]
            
            # Skip if importance is zero to avoid division by zero
            if ref_imp == 0:
                continue
                
            # Calculate DIS
            dis = ((curr_imp - ref_imp) / ref_imp) * 100
            
            # Determine if this feature had drift
            had_drift = (
                feature in drift_results and 
                drift_results[feature]["drift_detected"]
            )
            
            feature_impact[feature] = {
                "reference_importance": float(ref_imp),
                "current_importance": float(curr_imp), 
                "drift_impact_score": float(dis),
                "had_drift": had_drift,
                "abs_dis": abs(float(dis))
            }
        
        # Calculate Global DIS (weighted by reference importance)
        total_importance = sum(v["reference_importance"] for v in feature_impact.values())
        global_dis = 0
        
        if total_importance > 0:
            for feature, impact in feature_impact.items():
                weight = impact["reference_importance"] / total_importance
                # Use absolute DIS for global measure
                global_dis += weight * impact["abs_dis"]
                
        # Categorize features by impact severity
        high_impact_features = []
        medium_impact_features = []
        low_impact_features = []
        
        for feature, impact in feature_impact.items():
            if impact["had_drift"]:
                if impact["abs_dis"] > 50:  # >50% change in importance
                    high_impact_features.append(feature)
                elif impact["abs_dis"] > 20:  # >20% change
                    medium_impact_features.append(feature)
                else:
                    low_impact_features.append(feature)
                    
        return {
            "feature_impact": feature_impact,
            "global_dis": float(global_dis),
            "high_impact_features": high_impact_features,
            "medium_impact_features": medium_impact_features, 
            "low_impact_features": low_impact_features,
            "impact_summary": self._generate_impact_summary(
                global_dis, high_impact_features, medium_impact_features, feature_impact
            )
        }
        
    def _generate_impact_summary(self, global_dis: float, high_impact: List[str], 
                               medium_impact: List[str], 
                               feature_impact: Dict[str, Dict[str, Any]]) -> str:
        """Generate a human-readable summary of drift impact."""
        if global_dis < 5:
            risk_level = "very low"
        elif global_dis < 15:
            risk_level = "low"
        elif global_dis < 30:
            risk_level = "medium"
        elif global_dis < 50:
            risk_level = "high"
        else:
            risk_level = "critical"
            
        summary = [f"The overall drift impact is {risk_level} (Global DIS: {global_dis:.1f}%)."]
        
        if high_impact:
            # Get top 3 high impact features
            top_high_impact = sorted(
                [(f, feature_impact[f]["abs_dis"]) for f in high_impact],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            feature_summaries = []
            for feature, dis in top_high_impact:
                direction = "increased" if feature_impact[feature]["drift_impact_score"] > 0 else "decreased"
                feature_summaries.append(
                    f"{feature} ({direction} importance by {abs(feature_impact[feature]['drift_impact_score']):.1f}%)"
                )
            
            summary.append("High impact drift detected in features: " + ", ".join(feature_summaries))
        
        if not high_impact and not medium_impact:
            summary.append("No significant impact on model behavior detected.")
            
        return " ".join(summary)
