"""
Enhanced drift detection metrics for XAdapt-Drift framework.

This module implements additional statistical measures for drift detection
including KL divergence, Jensen-Shannon divergence, and other advanced metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, Any, Tuple, Optional
import warnings

class AdvancedDriftDetector:
    """Advanced drift detector with multiple statistical measures."""
    
    def __init__(self, bins: int = 50, alpha: float = 0.05):
        """
        Args:
            bins: Number of bins for histogram-based metrics
            alpha: Significance level for statistical tests
        """
        self.bins = bins
        self.alpha = alpha
    
    def kl_divergence(self, reference: np.ndarray, current: np.ndarray) -> float:
        """
        Calculate Kullback-Leibler divergence between two distributions.
        
        Args:
            reference: Reference data
            current: Current data
            
        Returns:
            KL divergence value
        """
        # Create histograms
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bins = np.linspace(min_val, max_val, self.bins + 1)
        
        ref_hist, _ = np.histogram(reference, bins=bins, density=True)
        curr_hist, _ = np.histogram(current, bins=bins, density=True)
        
        # Normalize to get probabilities
        ref_hist = ref_hist / ref_hist.sum()
        curr_hist = curr_hist / curr_hist.sum()
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        ref_hist = ref_hist + epsilon
        curr_hist = curr_hist + epsilon
        
        # Calculate KL divergence
        kl_div = np.sum(curr_hist * np.log(curr_hist / ref_hist))
        return kl_div
    
    def jensen_shannon_divergence(self, reference: np.ndarray, current: np.ndarray) -> float:
        """
        Calculate Jensen-Shannon divergence between two distributions.
        
        Args:
            reference: Reference data
            current: Current data
            
        Returns:
            JS divergence value
        """
        # Create histograms
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bins = np.linspace(min_val, max_val, self.bins + 1)
        
        ref_hist, _ = np.histogram(reference, bins=bins, density=True)
        curr_hist, _ = np.histogram(current, bins=bins, density=True)
        
        # Normalize to get probabilities
        ref_hist = ref_hist / ref_hist.sum()
        curr_hist = curr_hist / curr_hist.sum()
        
        # Calculate JS divergence
        js_div = jensenshannon(ref_hist, curr_hist) ** 2
        return js_div
    
    def categorical_kl_divergence(self, reference: pd.Series, current: pd.Series) -> float:
        """
        Calculate KL divergence for categorical data.
        
        Args:
            reference: Reference categorical data
            current: Current categorical data
            
        Returns:
            KL divergence value
        """
        # Get all unique categories
        all_categories = list(set(reference.unique()) | set(current.unique()))
        
        # Calculate frequencies
        ref_counts = reference.value_counts()
        curr_counts = current.value_counts()
        
        # Convert to probabilities
        ref_probs = np.array([ref_counts.get(cat, 0) for cat in all_categories])
        curr_probs = np.array([curr_counts.get(cat, 0) for cat in all_categories])
        
        ref_probs = ref_probs / ref_probs.sum()
        curr_probs = curr_probs / curr_probs.sum()
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        ref_probs = ref_probs + epsilon
        curr_probs = curr_probs + epsilon
        
        # Calculate KL divergence
        kl_div = np.sum(curr_probs * np.log(curr_probs / ref_probs))
        return kl_div
    
    def categorical_jensen_shannon_divergence(self, reference: pd.Series, current: pd.Series) -> float:
        """
        Calculate Jensen-Shannon divergence for categorical data.
        
        Args:
            reference: Reference categorical data
            current: Current categorical data
            
        Returns:
            JS divergence value
        """
        # Get all unique categories
        all_categories = list(set(reference.unique()) | set(current.unique()))
        
        # Calculate frequencies
        ref_counts = reference.value_counts()
        curr_counts = current.value_counts()
        
        # Convert to probabilities
        ref_probs = np.array([ref_counts.get(cat, 0) for cat in all_categories])
        curr_probs = np.array([curr_counts.get(cat, 0) for cat in all_categories])
        
        ref_probs = ref_probs / ref_probs.sum()
        curr_probs = curr_probs / curr_probs.sum()
        
        # Calculate JS divergence
        js_div = jensenshannon(ref_probs, curr_probs) ** 2
        return js_div
    
    def comprehensive_drift_analysis(self, reference: pd.DataFrame, 
                                   current: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Perform comprehensive drift analysis using multiple metrics.
        
        Args:
            reference: Reference dataframe
            current: Current dataframe
            
        Returns:
            Dictionary with comprehensive drift analysis results
        """
        results = {}
        
        for column in reference.columns:
            if column not in current.columns:
                continue
                
            ref_series = reference[column].dropna()
            curr_series = current[column].dropna()
            
            if len(ref_series) == 0 or len(curr_series) == 0:
                continue
            
            # Determine if numerical or categorical
            is_numerical = pd.api.types.is_numeric_dtype(ref_series)
            
            column_results = {
                'feature_type': 'numerical' if is_numerical else 'categorical'
            }
            
            if is_numerical:
                # Numerical feature analysis
                try:
                    # KL Divergence
                    kl_div = self.kl_divergence(ref_series.values, curr_series.values)
                    column_results['kl_divergence'] = kl_div
                    
                    # Jensen-Shannon Divergence
                    js_div = self.jensen_shannon_divergence(ref_series.values, curr_series.values)
                    column_results['jensen_shannon_divergence'] = js_div
                    
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_pvalue = stats.ks_2samp(ref_series, curr_series)
                    column_results['ks_statistic'] = ks_stat
                    column_results['ks_pvalue'] = ks_pvalue
                    column_results['ks_drift_detected'] = ks_pvalue < self.alpha
                    
                    # Wasserstein distance (Earth Mover's Distance)
                    wasserstein_dist = stats.wasserstein_distance(ref_series, curr_series)
                    column_results['wasserstein_distance'] = wasserstein_dist
                    
                    # Additional statistical measures
                    column_results['mean_shift'] = curr_series.mean() - ref_series.mean()
                    column_results['std_ratio'] = curr_series.std() / ref_series.std() if ref_series.std() > 0 else float('inf')
                    
                except Exception as e:
                    column_results['error'] = str(e)
                    
            else:
                # Categorical feature analysis
                try:
                    # KL Divergence for categorical
                    kl_div = self.categorical_kl_divergence(ref_series, curr_series)
                    column_results['kl_divergence'] = kl_div
                    
                    # Jensen-Shannon Divergence for categorical
                    js_div = self.categorical_jensen_shannon_divergence(ref_series, curr_series)
                    column_results['jensen_shannon_divergence'] = js_div
                    
                    # Chi-square test
                    # Create contingency table
                    all_categories = list(set(ref_series.unique()) | set(curr_series.unique()))
                    ref_counts = ref_series.value_counts()
                    curr_counts = curr_series.value_counts()
                    
                    observed = [curr_counts.get(cat, 0) for cat in all_categories]
                    expected = [ref_counts.get(cat, 0) for cat in all_categories]
                    
                    # Normalize expected to same total as observed
                    total_observed = sum(observed)
                    total_expected = sum(expected)
                    if total_expected > 0:
                        expected = [exp * total_observed / total_expected for exp in expected]
                    
                    # Perform chi-square test
                    chi2_stat, chi2_pvalue = stats.chisquare(observed, expected)
                    column_results['chi2_statistic'] = chi2_stat
                    column_results['chi2_pvalue'] = chi2_pvalue
                    column_results['chi2_drift_detected'] = chi2_pvalue < self.alpha
                    
                    # Category changes
                    ref_categories = set(ref_series.unique())
                    curr_categories = set(curr_series.unique())
                    
                    column_results['new_categories'] = list(curr_categories - ref_categories)
                    column_results['missing_categories'] = list(ref_categories - curr_categories)
                    column_results['category_count_change'] = len(curr_categories) - len(ref_categories)
                    
                except Exception as e:
                    column_results['error'] = str(e)
            
            # Overall drift decision based on multiple metrics
            drift_indicators = []
            if 'ks_drift_detected' in column_results:
                drift_indicators.append(column_results['ks_drift_detected'])
            if 'chi2_drift_detected' in column_results:
                drift_indicators.append(column_results['chi2_drift_detected'])
            
            # Add KL and JS divergence thresholds
            if 'kl_divergence' in column_results:
                kl_threshold = 0.1 if is_numerical else 0.05
                drift_indicators.append(column_results['kl_divergence'] > kl_threshold)
            
            if 'jensen_shannon_divergence' in column_results:
                js_threshold = 0.1 if is_numerical else 0.05
                drift_indicators.append(column_results['jensen_shannon_divergence'] > js_threshold)
            
            column_results['overall_drift_detected'] = any(drift_indicators) if drift_indicators else False
            column_results['drift_confidence'] = sum(drift_indicators) / len(drift_indicators) if drift_indicators else 0.0
            
            results[column] = column_results
            
        return results
    
    def create_drift_summary(self, analysis_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary of the drift analysis results.
        
        Args:
            analysis_results: Results from comprehensive_drift_analysis
            
        Returns:
            Summary dictionary
        """
        total_features = len(analysis_results)
        drifted_features = [f for f, r in analysis_results.items() if r.get('overall_drift_detected', False)]
        
        numerical_features = [f for f, r in analysis_results.items() if r.get('feature_type') == 'numerical']
        categorical_features = [f for f, r in analysis_results.items() if r.get('feature_type') == 'categorical']
        
        numerical_drifted = [f for f in drifted_features if f in numerical_features]
        categorical_drifted = [f for f in drifted_features if f in categorical_features]
        
        # Calculate average metrics
        avg_kl_div = np.mean([r.get('kl_divergence', 0) for r in analysis_results.values() if 'kl_divergence' in r])
        avg_js_div = np.mean([r.get('jensen_shannon_divergence', 0) for r in analysis_results.values() if 'jensen_shannon_divergence' in r])
        
        summary = {
            'total_features_analyzed': total_features,
            'drifted_features_count': len(drifted_features),
            'drift_ratio': len(drifted_features) / total_features if total_features > 0 else 0,
            'drifted_features': drifted_features,
            'numerical_features_count': len(numerical_features),
            'categorical_features_count': len(categorical_features),
            'numerical_drifted_count': len(numerical_drifted),
            'categorical_drifted_count': len(categorical_drifted),
            'average_kl_divergence': avg_kl_div,
            'average_js_divergence': avg_js_div,
            'high_confidence_drifts': [f for f, r in analysis_results.items() if r.get('drift_confidence', 0) > 0.8]
        }
        
        return summary


def demonstrate_advanced_drift_detection():
    """
    Demonstrate the advanced drift detection capabilities.
    """
    # Generate sample data with known drift
    np.random.seed(42)
    
    # Reference data
    n_samples = 1000
    ref_data = {
        'numerical_1': np.random.normal(0, 1, n_samples),
        'numerical_2': np.random.exponential(2, n_samples),
        'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2])
    }
    reference_df = pd.DataFrame(ref_data)
    
    # Current data with induced drift
    curr_data = {
        'numerical_1': np.random.normal(1.5, 1, n_samples),  # Mean shift
        'numerical_2': np.random.exponential(2, n_samples) * 2,  # Variance change
        'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.2, 0.3, 0.5])  # Frequency change
    }
    current_df = pd.DataFrame(curr_data)
    
    # Perform advanced drift detection
    detector = AdvancedDriftDetector(bins=50, alpha=0.05)
    results = detector.comprehensive_drift_analysis(reference_df, current_df)
    summary = detector.create_drift_summary(results)
    
    print("=== Advanced Drift Detection Results ===")
    print(f"Total features analyzed: {summary['total_features_analyzed']}")
    print(f"Features with drift detected: {summary['drifted_features_count']}")
    print(f"Drift ratio: {summary['drift_ratio']:.2%}")
    print(f"Average KL divergence: {summary['average_kl_divergence']:.4f}")
    print(f"Average JS divergence: {summary['average_js_divergence']:.4f}")
    
    print("\n=== Detailed Results by Feature ===")
    for feature, result in results.items():
        print(f"\n{feature} ({result['feature_type']}):")
        if result['feature_type'] == 'numerical':
            print(f"  KL Divergence: {result.get('kl_divergence', 'N/A'):.4f}")
            print(f"  JS Divergence: {result.get('jensen_shannon_divergence', 'N/A'):.4f}")
            print(f"  KS Test p-value: {result.get('ks_pvalue', 'N/A'):.4f}")
            print(f"  Wasserstein Distance: {result.get('wasserstein_distance', 'N/A'):.4f}")
            print(f"  Mean Shift: {result.get('mean_shift', 'N/A'):.4f}")
        else:
            print(f"  KL Divergence: {result.get('kl_divergence', 'N/A'):.4f}")
            print(f"  JS Divergence: {result.get('jensen_shannon_divergence', 'N/A'):.4f}")
            print(f"  Chi2 Test p-value: {result.get('chi2_pvalue', 'N/A'):.4f}")
            print(f"  New categories: {result.get('new_categories', [])}")
            print(f"  Missing categories: {result.get('missing_categories', [])}")
        
        print(f"  Overall drift detected: {result.get('overall_drift_detected', False)}")
        print(f"  Drift confidence: {result.get('drift_confidence', 0):.2f}")


if __name__ == "__main__":
    demonstrate_advanced_drift_detection()
