"""
XAdapt-Drift Basic Example

This example demonstrates how to use XAdapt-Drift to:
1. Detect drift between reference and current datasets
2. Characterize the nature of the drift
3. Analyze the impact on model behavior

The example uses a synthetic dataset with an induced drift on one feature.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Import XAdapt-Drift
from xadapt_drift import XAdaptDrift
from xadapt_drift.adapters.sklearn_adapter import SklearnAdapter

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
print("Generating synthetic dataset...")
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42
)

# Split into train (reference) and test (current) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create feature names
feature_names = [f'feature_{i}' for i in range(X.shape[1])]

# Convert to DataFrames
reference_df = pd.DataFrame(X_train, columns=feature_names)
current_df = pd.DataFrame(X_test, columns=feature_names)

# Train a model
print("Training a model on reference data...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Model accuracy on reference data: {model.score(X_train, y_train):.4f}")
print(f"Model accuracy on current data (before drift): {model.score(X_test, y_test):.4f}")

# Now let's introduce drift to one important feature
# First, find an important feature using feature importances
feature_importances = model.feature_importances_
most_important_idx = np.argmax(feature_importances)
most_important_feature = feature_names[most_important_idx]

print(f"\nIntroducing drift to most important feature: {most_important_feature}")
print(f"Original feature importance: {feature_importances[most_important_idx]:.4f}")

# Induce drift by shifting the mean
drift_magnitude = 1.5 * np.std(X_test[:, most_important_idx])
drifted_data = X_test.copy()
drifted_data[:, most_important_idx] += drift_magnitude

# Create drifted DataFrame
drifted_df = pd.DataFrame(drifted_data, columns=feature_names)

# Check model performance after drift
drifted_accuracy = model.score(drifted_data, y_test)
print(f"Model accuracy after drift: {drifted_accuracy:.4f}")
print(f"Performance drop: {(model.score(X_test, y_test) - drifted_accuracy) * 100:.2f}%")

# Now let's use XAdapt-Drift to analyze the drift
print("\nAnalyzing drift with XAdapt-Drift...")

# Create model adapter
model_adapter = SklearnAdapter(model, feature_names=feature_names)

# Create XAdapt-Drift analyzer
xadapt = XAdaptDrift(
    model_adapter=model_adapter,
    drift_threshold=0.05,
    explanation_method="permutation", 
    output_dir="./reports"
)

# Run analysis
report = xadapt.analyze(
    reference=reference_df,
    current=drifted_df,
    y_reference=y_train,
    y_current=y_test,
    report_name="synthetic_drift_report.json"
)

# Print results
print("\n--- XAdapt-Drift Analysis Results ---")
print("\nExecutive Summary:")
print(report["executive_summary"])

print("\nDrifted Features:")
for feature in report["drift_summary"]["drifted_features"]:
    print(f"- {feature}")
    
if report["characterization"]:
    print("\nCharacterization of Most Important Drifted Feature:")
    if most_important_feature in report["characterization"]:
        char = report["characterization"][most_important_feature]
        print(char["summary"])
        
        # Print detailed stats
        print(f"\nReference mean: {char['reference_stats']['mean']:.4f}")
        print(f"Current mean: {char['current_stats']['mean']:.4f}")
        print(f"Mean shift: {char['percent_diff']['mean']:.2f}%")
        
if "impact_analysis" in report:
    print("\nImpact Analysis:")
    print(f"Global DIS: {report['impact_analysis']['global_dis']:.2f}%")
    
    print("\nFeatures with High Impact:")
    for feature in report["impact_analysis"].get("high_impact_features", []):
        impact = report["impact_analysis"]["feature_impact"][feature]
        print(f"- {feature}: DIS = {impact['drift_impact_score']:.2f}% (Had drift: {impact['had_drift']})")

    print("\nFeatures with Medium Impact:")
    for feature in report["impact_analysis"].get("medium_impact_features", []):
        impact = report["impact_analysis"]["feature_impact"][feature]
        print(f"- {feature}: DIS = {impact['drift_impact_score']:.2f}% (Had drift: {impact['had_drift']})")

print("\nDone! Full report saved to ./reports/synthetic_drift_report.json")
