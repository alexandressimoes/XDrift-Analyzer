"""
Basic tests for XAdapt-Drift components
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from xadapt_drift import XAdaptDrift
from xadapt_drift.adapters.sklearn_adapter import SklearnAdapter
from xadapt_drift.drift.detector import DriftDetector
from xadapt_drift.drift.characterizer import DriftCharacterizer
from xadapt_drift.impact.analyzer import ImpactAnalyzer
from xadapt_drift.report.generator import ReportGenerator


@pytest.fixture
def synthetic_data():
    """Generate synthetic data for testing."""
    X, y = make_classification(
        n_samples=500, n_features=5, n_informative=3, random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Split into reference and current
    X_ref, X_curr = X[:400], X[400:]
    y_ref, y_curr = y[:400], y[400:]
    
    # Create DataFrames
    reference_df = pd.DataFrame(X_ref, columns=feature_names)
    current_df = pd.DataFrame(X_curr, columns=feature_names)
    
    return reference_df, current_df, y_ref, y_curr, feature_names


def test_drift_detector(synthetic_data):
    """Test that drift detector works properly."""
    reference_df, current_df, _, _, _ = synthetic_data
    
    # Create detector
    detector = DriftDetector(threshold=0.05)
    
    # Run detection
    results = detector.detect(reference_df, current_df)
    
    # Assertions
    assert isinstance(results, dict)
    assert len(results) == reference_df.shape[1]
    for feature, result in results.items():
        assert "drift_detected" in result
        assert "p_value" in result
        assert "statistic" in result
        assert "method" in result
        assert "feature_type" in result


def test_drift_characterizer(synthetic_data):
    """Test that drift characterizer works properly."""
    reference_df, current_df, _, _, _ = synthetic_data
    
    # Create detector and characterizer
    detector = DriftDetector(threshold=0.05)
    characterizer = DriftCharacterizer()
    
    # Run detection
    drift_results = detector.detect(reference_df, current_df)
    
    # Force at least one feature to have drift
    for feature in drift_results:
        drift_results[feature]["drift_detected"] = True
        break
    
    # Run characterization
    char_results = characterizer.characterize(drift_results, reference_df, current_df)
    
    # Assertions
    assert isinstance(char_results, dict)
    for feature, result in char_results.items():
        assert drift_results[feature]["drift_detected"]
        if drift_results[feature]["feature_type"] == "numerical":
            assert "reference_stats" in result
            assert "current_stats" in result
            assert "absolute_diff" in result
            assert "percent_diff" in result
            assert "major_changes" in result
            assert "summary" in result
        else:
            assert "category_changes" in result
            assert "new_categories" in result
            assert "missing_categories" in result
            assert "significant_changes" in result
            assert "summary" in result


def test_impact_analyzer(synthetic_data):
    """Test that impact analyzer works properly."""
    reference_df, current_df, y_ref, y_curr, feature_names = synthetic_data
    
    # Train a model
    model = RandomForestClassifier(random_state=42)
    model.fit(reference_df, y_ref)
    
    # Create model adapter
    model_adapter = SklearnAdapter(model, feature_names=feature_names)
    
    # Create detector
    detector = DriftDetector(threshold=0.05)
    
    # Run detection
    drift_results = detector.detect(reference_df, current_df)
    
    # Force at least one feature to have drift
    for feature in drift_results:
        drift_results[feature]["drift_detected"] = True
        break
    
    # Create impact analyzer
    analyzer = ImpactAnalyzer()
    
    # Run impact analysis
    impact_results = analyzer.analyze(
        drift_results, reference_df, current_df, model_adapter, 
        y_ref, y_curr, "permutation"
    )
    
    # Assertions
    assert isinstance(impact_results, dict)
    assert "feature_impact" in impact_results
    assert "global_dis" in impact_results
    assert isinstance(impact_results["global_dis"], float)
    assert "high_impact_features" in impact_results
    assert "medium_impact_features" in impact_results
    assert "low_impact_features" in impact_results
    assert "impact_summary" in impact_results


def test_xadapt_drift_integration(synthetic_data):
    """Test that the full XAdapt-Drift integration works properly."""
    reference_df, current_df, y_ref, y_curr, feature_names = synthetic_data
    
    # Train a model
    model = RandomForestClassifier(random_state=42)
    model.fit(reference_df, y_ref)
    
    # Create model adapter
    model_adapter = SklearnAdapter(model, feature_names=feature_names)
    
    # Create XAdapt-Drift
    xadapt = XAdaptDrift(
        model_adapter=model_adapter,
        drift_threshold=0.05,
        explanation_method="permutation"
    )
    
    # Run analysis
    report = xadapt.analyze(
        reference=reference_df,
        current=current_df,
        y_reference=y_ref,
        y_current=y_curr
    )
    
    # Assertions
    assert isinstance(report, dict)
    assert "executive_summary" in report
    assert "drift_summary" in report
    assert "drift_details" in report
    assert "characterization" in report
    assert "impact_analysis" in report
    assert "metadata" in report
