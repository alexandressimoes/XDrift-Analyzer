"""
XAdapt-Drift Advanced Example

This example demonstrates more advanced features of XAdapt-Drift:
1. Handling mixed data types (numerical and categorical features)
2. Visualizing drift characterization
3. Comparing drift impact with actual performance degradation
"""

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

# Import XAdapt-Drift
from xadapt_drift import XAdaptDrift
from xadapt_drift.adapters.sklearn_adapter import SklearnAdapter

# Set up plotting style
plt.style.use('seaborn-whitegrid')
sns.set_palette('viridis')

def create_synthetic_data(n_samples=10000, n_cat_features=3, n_num_features=5):
    """Create a synthetic dataset with mixed data types.
    Args:
        n_samples: Total number of samples
        n_cat_features: Number of categorical features
        n_num_features: Number of numerical features
    Returns:
        DataFrame with mixed features and target variable
        y.astype(int): Binary target variable
        num_cols: List of numerical feature names
        cat_cols: List of categorical feature names
    """
    # Create numerical features
    X_num = np.random.randn(n_samples, n_num_features)
    
    # Create categorical features (3 categories each)
    X_cat = np.random.randint(0, 3, size=(n_samples, n_cat_features))
    
    # Create target based on both numerical and categorical features
    y = (0.5 * np.sum(X_num[:, :2], axis=1) + 
         0.8 * (X_cat[:, 0] == 2).astype(int) - 
         0.5 * (X_cat[:, 1] == 0).astype(int) + 
         0.1 * np.random.randn(n_samples)) > 0
    
    # Combine features
    X = np.hstack([X_num, X_cat])
    
    # Create feature names
    num_cols = [f'num_{i}' for i in range(n_num_features)]
    cat_cols = [f'cat_{i}' for i in range(n_cat_features)]
    feature_names = num_cols + cat_cols
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Convert categorical columns to correct type
    for col in cat_cols:
        df[col] = df[col].astype('category')
    
    return df, y.astype(int), num_cols, cat_cols

def induce_drift(df, num_cols, cat_cols, drift_type='mean_shift'):
    """Induce different types of drift in the dataset.
    Args:
        df: Original DataFrame
        num_cols: List of numerical feature names
        cat_cols: List of categorical feature names
        drift_type: Type of drift to induce ('mean_shift', 'variance_change', 'category_frequency', 'multiple')
    Returns:
        drifted_df: DataFrame with induced drift
        drifted_features: List of features that were changed
    """
    
    drifted_df = df.copy()
    
    if drift_type == 'mean_shift':
        # Shift the mean of the first numerical feature
        feature = num_cols[0]
        shift = 1.5 * drifted_df[feature].std()
        drifted_df[feature] += shift
        drifted_features = [feature]
        
    elif drift_type == 'variance_change':
        # Increase the variance of the second numerical feature
        feature = num_cols[1]
        drifted_df[feature] = drifted_df[feature] * 2.0
        drifted_features = [feature]
    
    elif drift_type == 'category_frequency':
        # Change the distribution of a categorical feature
        feature = cat_cols[0]
        # Find the least common category
        least_common = drifted_df[feature].value_counts().idxmin()
        # Make it more common by replacing some values
        mask = np.random.choice([True, False], size=len(drifted_df), p=[0.4, 0.6])
        drifted_df.loc[mask, feature] = least_common
        drifted_features = [feature]
        
    elif drift_type == 'multiple':
        # Induce multiple drifts
        # Shift mean of first numerical feature
        drifted_df[num_cols[0]] += 1.2 * drifted_df[num_cols[0]].std()
        # Increase variance of second numerical feature
        drifted_df[num_cols[1]] = drifted_df[num_cols[1]] * 1.8
        # Change categorical distribution
        feature = cat_cols[0]
        mask = np.random.choice([True, False], size=len(drifted_df), p=[0.3, 0.7])
        drifted_df.loc[mask, feature] = drifted_df[feature].value_counts().idxmin()
        drifted_features = [num_cols[0], num_cols[1], cat_cols[0]]
    
    return drifted_df, drifted_features

def visualize_numerical_drift(reference_df, current_df, feature, figsize=(10, 6)):
    """Visualize drift in numerical features using KDE plots."""
    plt.figure(figsize=figsize)
    
    sns.kdeplot(reference_df[feature], label='Reference', fill=True, alpha=0.3)
    sns.kdeplot(current_df[feature], label='Current', fill=True, alpha=0.3)
    
    plt.title(f'Distribution Shift in {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

def visualize_categorical_drift(reference_df, current_df, feature, figsize=(10, 6)):
    """Visualize drift in categorical features using bar plots."""
    plt.figure(figsize=figsize)
    
    ref_counts = reference_df[feature].value_counts(normalize=True)
    curr_counts = current_df[feature].value_counts(normalize=True)
    
    # Ensure all categories are present in both
    all_cats = sorted(set(ref_counts.index) | set(curr_counts.index))
    
    x = np.arange(len(all_cats))
    width = 0.35
    
    ref_values = [ref_counts.get(cat, 0) for cat in all_cats]
    curr_values = [curr_counts.get(cat, 0) for cat in all_cats]
    
    plt.bar(x - width/2, ref_values, width, label='Reference')
    plt.bar(x + width/2, curr_values, width, label='Current')
    
    plt.xlabel('Category')
    plt.ylabel('Frequency')
    plt.title(f'Frequency Shift in {feature}')
    plt.xticks(x, all_cats)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

def visualize_impact_vs_performance(drift_impacts, performance_drops, feature_names, figsize=(12, 7)):
    """Visualize correlation between drift impact scores and performance drop."""
    plt.figure(figsize=figsize)
    
    # Create scatter plot
    plt.scatter(drift_impacts, performance_drops, s=80, alpha=0.7)
    
    # Add feature labels
    for i, feature in enumerate(feature_names):
        plt.annotate(feature, (drift_impacts[i], performance_drops[i]), 
                     xytext=(7, 3), textcoords='offset points')
    
    # Add trend line
    z = np.polyfit(drift_impacts, performance_drops, 1)
    p = np.poly1d(z)
    plt.plot(drift_impacts, p(drift_impacts), "r--", alpha=0.8)
    
    plt.xlabel('Drift Impact Score (absolute %)')
    plt.ylabel('Performance Drop (%)')
    plt.title('Correlation between Drift Impact and Model Performance')
    plt.grid(True)
    plt.tight_layout()
    
    # Calculate correlation
    corr = np.corrcoef(drift_impacts, performance_drops)[0, 1]
    plt.figtext(0.15, 0.85, f"Correlation: {corr:.2f}", fontsize=12)
    
    return plt.gcf()

def main():
    print("=== XAdapt-Drift Advanced Example ===")
    
    # Create synthetic data
    print("\nGenerating synthetic mixed-type dataset...")
    reference_df, y_ref, num_cols, cat_cols = create_synthetic_data(n_samples=2000)
    print(f"Created dataset with {len(num_cols)} numerical features and {len(cat_cols)} categorical features")
    
    # Split into training and test sets
    train_idx = np.random.choice([True, False], size=len(reference_df), p=[0.7, 0.3])
    train_df = reference_df[train_idx].copy()
    test_df = reference_df[~train_idx].copy()
    y_train = y_ref[train_idx]
    y_test = y_ref[~train_idx]
    
    # Create a preprocessing pipeline for mixed data types
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )
    
    # Create and train model
    print("\nTraining model...")
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    model.fit(train_df, y_train)
    
    # Calculate baseline accuracy
    baseline_acc = accuracy_score(y_test, model.predict(test_df))
    baseline_auc = roc_auc_score(y_test, model.predict_proba(test_df)[:, 1])
    
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    print(f"Baseline AUC: {baseline_auc:.4f}")
    
    # Create a model adapter
    # For simplicity, we'll extract the classifier from the pipeline
    classifier = model.named_steps['classifier']
    # We need to know the feature names after preprocessing
    # For this example, we'll use the original feature names, but note that 
    # in a real scenario with OneHotEncoding, the feature names would change
    model_adapter = SklearnAdapter(classifier, feature_names=num_cols + cat_cols)
    
    # Setup XAdapt-Drift
    xadapt = XAdaptDrift(
        model_adapter=model_adapter,
        drift_threshold=0.05,
        explanation_method="permutation"
    )
    
    # Test different drift scenarios
    drift_types = ['mean_shift', 'variance_change', 'category_frequency', 'multiple']
    
    results = []
    
    for drift_type in drift_types:
        print(f"\n--- Testing {drift_type} drift ---")
        
        # Induce drift
        drifted_df, drifted_features = induce_drift(test_df, num_cols, cat_cols, drift_type)
        
        # Calculate performance after drift
        drifted_acc = accuracy_score(y_test, model.predict(drifted_df))
        drifted_auc = roc_auc_score(y_test, model.predict_proba(drifted_df)[:, 1])
        
        performance_drop_acc = (baseline_acc - drifted_acc) * 100
        performance_drop_auc = (baseline_auc - drifted_auc) * 100
        
        print(f"Accuracy after drift: {drifted_acc:.4f} (drop: {performance_drop_acc:.2f}%)")
        print(f"AUC after drift: {drifted_auc:.4f} (drop: {performance_drop_auc:.2f}%)")
        
        # Analyze with XAdapt-Drift
        report = xadapt.analyze(
            reference=train_df,
            current=drifted_df,
            y_reference=y_train,
            y_current=y_test
        )
        
        # Print summary
        print("\nXAdapt-Drift Analysis:")
        print(report["executive_summary"])
        
        # Visualize one numerical and one categorical feature drift
        if len(set(drifted_features) & set(num_cols)) > 0:
            num_feature = list(set(drifted_features) & set(num_cols))[0]
            visualize_numerical_drift(train_df, drifted_df, num_feature)
            plt.savefig(f"numerical_drift_{drift_type}.png")
            
        if len(set(drifted_features) & set(cat_cols)) > 0:
            cat_feature = list(set(drifted_features) & set(cat_cols))[0]
            visualize_categorical_drift(train_df, drifted_df, cat_feature)
            plt.savefig(f"categorical_drift_{drift_type}.png")
            
        # Store results for later correlation analysis
        if "impact_analysis" in report:
            for feature in report["impact_analysis"]["feature_impact"]:
                impact = report["impact_analysis"]["feature_impact"][feature]
                results.append({
                    'drift_type': drift_type,
                    'feature': feature,
                    'had_drift': impact['had_drift'],
                    'drift_impact_score': abs(impact['drift_impact_score']),
                    'performance_drop_acc': performance_drop_acc,
                    'performance_drop_auc': performance_drop_auc,
                    'global_dis': report['impact_analysis']['global_dis']
                })
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Only analyze features that had drift
    drifted_results = results_df[results_df['had_drift']].copy()
    
    # Visualize correlation between drift impact and performance drop
    if len(drifted_results) > 0:
        print("\nAnalyzing correlation between drift impact and performance drop...")
        
        visualize_impact_vs_performance(
            drifted_results['drift_impact_score'].values,
            drifted_results['performance_drop_acc'].values,
            drifted_results['feature'].values
        )
        plt.savefig("impact_vs_performance.png")
        
        # Calculate correlation
        corr = np.corrcoef(
            drifted_results['drift_impact_score'].values,
            drifted_results['performance_drop_acc'].values
        )[0, 1]
        
        print(f"Correlation between DIS and accuracy drop: {corr:.4f}")
        
        # Check correlation between Global DIS and performance drop
        global_dis_corr = np.corrcoef(
            results_df.groupby('drift_type')['global_dis'].first().values,
            results_df.groupby('drift_type')['performance_drop_acc'].first().values
        )[0, 1]
        
        print(f"Correlation between Global DIS and accuracy drop: {global_dis_corr:.4f}")
    
    print("\nExample completed! Visualizations saved as PNG files.")

if __name__ == "__main__":
    main()
