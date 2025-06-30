import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any
from scipy import stats

class DriftDetector:
    """Detects data drift between reference and current data distributions."""
    
    def __init__(self, threshold: float = 0.05):
        """
        Args:
            threshold: p-value threshold for drift detection
        """
        self.threshold = threshold
        
    def detect(self, reference: pd.DataFrame, current: pd.DataFrame, 
               features: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Detect drift between reference and current data.
        
        Args:
            reference: Reference data
            current: Current data to check for drift
            features: Features to check (if None, checks all common features)
            
        Returns:
            Dictionary with drift results for each feature
        """
        if features is None:
            features = [col for col in reference.columns if col in current.columns]
            
        results = {}
        
        for feature in features:
            # Skip features with missing values
            if (reference[feature].isna().any() or current[feature].isna().any()):
                continue
                
            # Determine feature type (categorical or numerical)
            feature_type = self._determine_type(reference[feature])
            
            if feature_type == "categorical":
                result = self._detect_categorical_drift(reference[feature], current[feature])
            else:
                result = self._detect_numerical_drift(reference[feature], current[feature])
                
            results[feature] = {
                "drift_detected": result["p_value"] < self.threshold,
                "p_value": result["p_value"],
                "statistic": result["statistic"],
                "method": result["method"],
                "feature_type": feature_type
            }
            
        return results
    
    def _determine_type(self, series: pd.Series) -> str:
        """Determine if a feature is categorical or numerical."""
        if series.dtype.kind in 'iufc':  # integer, unsigned int, float, complex
            # Check if it's actually categorical (few unique values)
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.05:  # Fewer than 5% unique values
                return "categorical"
            return "numerical"
        return "categorical"
        
    def _detect_numerical_drift(self, reference: pd.Series, current: pd.Series) -> Dict[str, Any]:
        """Detect drift in numerical features using Kolmogorov-Smirnov test."""
        statistic, p_value = stats.ks_2samp(reference, current)
        return {
            "p_value": p_value,
            "statistic": statistic,
            "method": "kolmogorov_smirnov"
        }
    
    def _detect_categorical_drift(self, reference: pd.Series, current: pd.Series) -> Dict[str, Any]:
        """Detect drift in categorical features using Chi-Square test."""
        # Get value counts
        ref_counts = reference.value_counts(normalize=True)
        curr_counts = current.value_counts(normalize=True)
        
        # Combine categories
        all_categories = list(set(ref_counts.index) | set(curr_counts.index))
        
        # Create contingency table
        ref_vector = np.array([ref_counts.get(cat, 0) for cat in all_categories])
        curr_vector = np.array([curr_counts.get(cat, 0) for cat in all_categories])
        
        # Chi-square test
        statistic, p_value = stats.chisquare(curr_vector, ref_vector)
        
        return {
            "p_value": p_value,
            "statistic": statistic,
            "method": "chi_square"
        }
