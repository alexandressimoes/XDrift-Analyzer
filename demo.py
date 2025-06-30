"""
Quick test script for XAdapt-Drift framework.

This script creates a simple synthetic dataset and demonstrates the main functionality
of XAdapt-Drift on a synthetic drift scenario.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from xadapt_drift import XAdaptDrift
from xadapt_drift.adapters.sklearn_adapter import SklearnAdapter
from xadapt_drift.utils import create_drift_dashboard

def main():
    print("XAdapt-Drift Demo")
    print("=================")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    X, y = make_classification(
        n_samples=500,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42
    )
    
    # Create feature names and convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Split into reference and current
    reference_df, current_df, y_ref, y_curr = train_test_split(
        df, y, test_size=0.3, random_state=42
    )
    
    # Train a simple model
    print("Training a model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(reference_df, y_ref)
    print(f"Base model accuracy: {model.score(current_df, y_curr):.4f}")
    
    # Induce drift in an important feature
    print("Inducing drift in important feature...")
    feature_importances = model.feature_importances_
    most_important_idx = np.argmax(feature_importances)
    most_important_feature = feature_names[most_important_idx]
    
    # Copy the current data and induce drift
    drifted_df = current_df.copy()
    drifted_df[most_important_feature] += 1.0 * drifted_df[most_important_feature].std()
    
    # Check accuracy after drift
    drifted_accuracy = model.score(drifted_df, y_curr)
    print(f"Accuracy after drift: {drifted_accuracy:.4f}")
    print(f"Performance drop: {(model.score(current_df, y_curr) - drifted_accuracy) * 100:.2f}%")
    
    # Create model adapter
    model_adapter = SklearnAdapter(model, feature_names=feature_names)
    
    # Create XAdapt-Drift analyzer
    print("\nAnalyzing drift with XAdapt-Drift...")
    xadapt = XAdaptDrift(
        model_adapter=model_adapter,
        explanation_method="permutation"
    )
    
    # Run analysis
    report = xadapt.analyze(
        reference=reference_df,
        current=drifted_df,
        y_reference=y_ref,
        y_current=y_curr
    )
    
    # Print results
    print("\nExecutive Summary:")
    print(report["executive_summary"])
    
    print("\nDrifted Features:")
    for feature in report["drift_summary"]["drifted_features"]:
        print(f"- {feature}")
    
    # Show impact analysis
    if "impact_analysis" in report:
        print("\nImpact Analysis:")
        print(f"Global DIS: {report['impact_analysis']['global_dis']:.2f}%")
        
        # Check if the most important feature was identified correctly
        if most_important_feature in report["impact_analysis"]["high_impact_features"]:
            print(f"\nSuccessfully identified {most_important_feature} as high impact!")
            
    # Generate a dashboard
    print("\nGenerating visual dashboard...")
    dashboard = create_drift_dashboard(report, reference_df, drifted_df, "drift_dashboard.png")
    
    print("\nDemo completed! Dashboard saved as 'drift_dashboard.png'")

if __name__ == "__main__":
    main()
