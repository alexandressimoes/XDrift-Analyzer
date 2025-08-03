"""
Advanced example showing how BaseAdapter enables multi-framework drift analysis.

This example demonstrates how the adapter pattern allows the drift detection
system to work with models from different ML frameworks seamlessly.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from abc import ABC
import logging

from xadapt_drift.adapters.base import BaseAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiFrameworkDriftAnalyzer:
    """Drift analyzer that works with multiple ML frameworks via adapters."""
    
    def __init__(self, adapters: List[BaseAdapter]):
        """Initialize with a list of adapters from different frameworks.
        
        Args:
            adapters: List of BaseAdapter instances from different frameworks
        """
        self.adapters = adapters
        logger.info(f"Initialized analyzer with {len(adapters)} adapters")
        
        # Validate that all adapters have consistent feature names
        self._validate_adapter_compatibility()
    
    def _validate_adapter_compatibility(self) -> None:
        """Ensure all adapters have compatible feature sets."""
        if not self.adapters:
            raise ValueError("At least one adapter is required")
        
        # Get reference feature names from first adapter
        reference_features = set(self.adapters[0].feature_names)
        
        for i, adapter in enumerate(self.adapters[1:], 1):
            adapter_features = set(adapter.feature_names)
            if adapter_features != reference_features:
                logger.warning(
                    f"Adapter {i} has different features. "
                    f"Expected: {reference_features}, Got: {adapter_features}"
                )
    
    def analyze_prediction_drift(self, X_reference: np.ndarray, 
                                X_current: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction drift across multiple models.
        
        Args:
            X_reference: Reference dataset
            X_current: Current dataset to compare
            
        Returns:
            Dictionary with drift analysis results
        """
        logger.info("Starting multi-framework prediction drift analysis")
        
        results = {
            "framework_predictions": {},
            "prediction_drift_scores": {},
            "consensus_metrics": {}
        }
        
        # Get predictions from all adapters
        for adapter in self.adapters:
            framework_name = adapter.__class__.__name__.replace("Adapter", "")
            
            try:
                # Get predictions for both datasets
                pred_ref = adapter.predict(X_reference)
                pred_curr = adapter.predict(X_current)
                
                results["framework_predictions"][framework_name] = {
                    "reference": pred_ref,
                    "current": pred_curr
                }
                
                # Calculate basic drift metrics
                drift_score = self._calculate_prediction_drift(pred_ref, pred_curr)
                results["prediction_drift_scores"][framework_name] = drift_score
                
                logger.info(f"✅ {framework_name} analysis complete. Drift score: {drift_score:.4f}")
                
            except Exception as e:
                logger.error(f"❌ {framework_name} analysis failed: {e}")
                results["prediction_drift_scores"][framework_name] = None
        
        # Calculate consensus metrics
        valid_scores = [score for score in results["prediction_drift_scores"].values() 
                       if score is not None]
        
        if valid_scores:
            results["consensus_metrics"] = {
                "mean_drift_score": np.mean(valid_scores),
                "std_drift_score": np.std(valid_scores),
                "max_drift_score": np.max(valid_scores),
                "min_drift_score": np.min(valid_scores),
                "consensus_agreement": self._calculate_consensus_agreement(valid_scores)
            }
        
        return results
    
    def analyze_feature_importance_drift(self, X_reference: np.ndarray, 
                                       y_reference: np.ndarray,
                                       X_current: np.ndarray,
                                       y_current: np.ndarray) -> Dict[str, Any]:
        """Analyze feature importance drift across frameworks.
        
        Args:
            X_reference: Reference features
            y_reference: Reference targets
            X_current: Current features  
            y_current: Current targets
            
        Returns:
            Dictionary with feature importance drift analysis
        """
        logger.info("Starting multi-framework feature importance drift analysis")
        
        results = {
            "framework_importances": {},
            "importance_drift_scores": {},
            "feature_consensus": {}
        }
        
        # Get feature importances from all adapters
        for adapter in self.adapters:
            framework_name = adapter.__class__.__name__.replace("Adapter", "")
            
            try:
                # Get explanations for both datasets
                if hasattr(adapter, 'explain'):
                    # Try different explanation methods based on adapter type
                    if "TensorFlow" in framework_name:
                        method = "integrated_gradients"
                    elif "XGBoost" in framework_name:
                        method = "tree_shap"
                    else:
                        method = "shap"
                    
                    importance_ref = adapter.explain(X_reference, y_reference, method=method)
                    importance_curr = adapter.explain(X_current, y_current, method=method)
                    
                    results["framework_importances"][framework_name] = {
                        "reference": importance_ref,
                        "current": importance_curr
                    }
                    
                    # Calculate importance drift
                    drift_score = self._calculate_importance_drift(importance_ref, importance_curr)
                    results["importance_drift_scores"][framework_name] = drift_score
                    
                    logger.info(f"✅ {framework_name} importance analysis complete")
                
            except Exception as e:
                logger.error(f"❌ {framework_name} importance analysis failed: {e}")
                results["importance_drift_scores"][framework_name] = None
        
        # Calculate feature-level consensus
        results["feature_consensus"] = self._calculate_feature_consensus(
            results["framework_importances"]
        )
        
        return results
    
    def _calculate_prediction_drift(self, pred_ref: np.ndarray, 
                                  pred_curr: np.ndarray) -> float:
        """Calculate drift score between prediction distributions."""
        try:
            from scipy import stats
            # Use Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(pred_ref, pred_curr)
            return statistic
        except ImportError:
            # Fallback to simple metric
            return abs(np.mean(pred_ref) - np.mean(pred_curr))
    
    def _calculate_importance_drift(self, imp_ref: Dict[str, float], 
                                  imp_curr: Dict[str, float]) -> float:
        """Calculate drift in feature importances."""
        common_features = set(imp_ref.keys()) & set(imp_curr.keys())
        
        if not common_features:
            return 1.0  # Maximum drift if no common features
        
        diffs = []
        for feature in common_features:
            diff = abs(imp_ref[feature] - imp_curr[feature])
            diffs.append(diff)
        
        return np.mean(diffs)
    
    def _calculate_consensus_agreement(self, scores: List[float]) -> float:
        """Calculate how much frameworks agree on drift scores."""
        if len(scores) < 2:
            return 1.0
        
        # Calculate coefficient of variation (inverse of agreement)
        std_dev = np.std(scores)
        mean_score = np.mean(scores)
        
        if mean_score == 0:
            return 1.0
        
        cv = std_dev / mean_score
        agreement = 1.0 / (1.0 + cv)  # Higher agreement = lower CV
        return agreement
    
    def _calculate_feature_consensus(self, framework_importances: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate consensus on which features are most important."""
        if not framework_importances:
            return {}
        
        # Get all features
        all_features = set()
        for framework_data in framework_importances.values():
            if framework_data and "current" in framework_data:
                all_features.update(framework_data["current"].keys())
        
        feature_rankings = {}
        
        # Rank features by importance for each framework
        for framework, data in framework_importances.items():
            if data and "current" in data:
                # Sort features by importance
                sorted_features = sorted(
                    data["current"].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                feature_rankings[framework] = [f[0] for f in sorted_features]
        
        # Calculate rank correlation between frameworks
        consensus_score = self._calculate_rank_correlation(feature_rankings)
        
        return {
            "feature_rankings": feature_rankings,
            "consensus_score": consensus_score,
            "total_features": len(all_features)
        }
    
    def _calculate_rank_correlation(self, rankings: Dict[str, List[str]]) -> float:
        """Calculate average rank correlation between frameworks."""
        if len(rankings) < 2:
            return 1.0
        
        try:
            from scipy.stats import spearmanr
            
            frameworks = list(rankings.keys())
            correlations = []
            
            for i in range(len(frameworks)):
                for j in range(i + 1, len(frameworks)):
                    rank1 = rankings[frameworks[i]]
                    rank2 = rankings[frameworks[j]]
                    
                    # Get common features and their ranks
                    common_features = set(rank1) & set(rank2)
                    if len(common_features) > 1:
                        ranks1 = [rank1.index(f) for f in common_features]
                        ranks2 = [rank2.index(f) for f in common_features]
                        
                        corr, _ = spearmanr(ranks1, ranks2)
                        if not np.isnan(corr):
                            correlations.append(corr)
            
            return np.mean(correlations) if correlations else 0.0
            
        except ImportError:
            return 0.5  # Fallback when scipy not available
    
    def generate_report(self, prediction_results: Dict[str, Any], 
                       importance_results: Dict[str, Any]) -> str:
        """Generate a comprehensive drift analysis report."""
        
        report = []
        report.append("🔍 Multi-Framework Drift Analysis Report")
        report.append("=" * 50)
        
        # Prediction drift summary
        report.append("\n📊 Prediction Drift Analysis:")
        if prediction_results.get("consensus_metrics"):
            metrics = prediction_results["consensus_metrics"]
            report.append(f"   • Mean Drift Score: {metrics['mean_drift_score']:.4f}")
            report.append(f"   • Framework Agreement: {metrics['consensus_agreement']:.4f}")
            report.append(f"   • Score Range: {metrics['min_drift_score']:.4f} - {metrics['max_drift_score']:.4f}")
        
        # Feature importance drift summary
        report.append("\n🎯 Feature Importance Drift Analysis:")
        if importance_results.get("feature_consensus"):
            consensus = importance_results["feature_consensus"]
            report.append(f"   • Feature Consensus Score: {consensus['consensus_score']:.4f}")
            report.append(f"   • Total Features Analyzed: {consensus['total_features']}")
        
        # Framework-specific results
        report.append("\n🔧 Framework-Specific Results:")
        for framework, score in prediction_results.get("prediction_drift_scores", {}).items():
            if score is not None:
                report.append(f"   • {framework}: {score:.4f}")
            else:
                report.append(f"   • {framework}: Analysis Failed")
        
        return "\n".join(report)


# Example usage demonstration
def demonstrate_multi_framework_analysis():
    """Demonstrate multi-framework drift analysis."""
    
    # Import mock adapters from our demo
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    from adapter_pattern_demo import MockTensorFlowAdapter, MockXGBoostAdapter
    
    # Create sample data
    np.random.seed(42)
    n_samples_ref, n_samples_curr = 1000, 800
    n_features = 10
    
    X_reference = np.random.normal(0, 1, (n_samples_ref, n_features))
    X_current = np.random.normal(0.5, 1.2, (n_samples_curr, n_features))  # Shifted distribution
    
    y_reference = np.random.randint(0, 2, n_samples_ref)
    y_current = np.random.randint(0, 2, n_samples_curr)
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Create adapters for different frameworks
    adapters = [
        MockTensorFlowAdapter(object(), feature_names=feature_names),
        MockXGBoostAdapter(object(), feature_names=feature_names)
    ]
    
    # Initialize analyzer
    analyzer = MultiFrameworkDriftAnalyzer(adapters)
    
    # Run analysis
    print("🚀 Running Multi-Framework Drift Analysis...")
    
    prediction_results = analyzer.analyze_prediction_drift(X_reference, X_current)
    importance_results = analyzer.analyze_feature_importance_drift(
        X_reference, y_reference, X_current, y_current
    )
    
    # Generate and print report
    report = analyzer.generate_report(prediction_results, importance_results)
    print(report)
    
    return prediction_results, importance_results


if __name__ == "__main__":
    demonstrate_multi_framework_analysis()
