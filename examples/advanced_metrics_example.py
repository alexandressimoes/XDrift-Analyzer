"""
Example demonstrating advanced drift metrics integration with XAdapt-Drift.

This example shows how to use KL divergence, Jensen-Shannon divergence, 
and other advanced metrics alongside the standard XAdapt-Drift framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Import XAdapt-Drift components
from xadapt_drift import XAdaptDrift
from xadapt_drift.adapters.sklearn_adapter import SklearnAdapter
from xadapt_drift.utils.advanced_metrics import AdvancedDriftDetector

def create_enhanced_drift_scenarios():
    """Create different types of drift scenarios for testing advanced metrics."""
    
    # Generate base dataset
    X, y = make_classification(
        n_samples=2000,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_clusters_per_class=2,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Add categorical features
    categorical_data = np.random.choice(['Type_A', 'Type_B', 'Type_C'], 
                                      size=(X.shape[0], 2), 
                                      p=[0.5, 0.3, 0.2])
    
    # Combine numerical and categorical
    full_data = np.hstack([X, categorical_data])
    feature_names.extend(['category_1', 'category_2'])
    
    # Create DataFrame
    df = pd.DataFrame(full_data, columns=feature_names)
    
    # Convert categorical columns
    for col in ['category_1', 'category_2']:
        df[col] = df[col].astype('category')
    
    # Split into reference and test
    reference_df, test_df, y_ref, y_test = train_test_split(
        df, y, test_size=0.4, random_state=42
    )
    
    return reference_df, test_df, y_ref, y_test, feature_names

def induce_specific_drifts(test_df, feature_names):
    """Induce specific types of drift for testing different metrics."""
    
    scenarios = {}
    
    # Scenario 1: Gradual mean shift (detectable by KL/JS divergence)
    scenario_1 = test_df.copy()
    numerical_features = [f for f in feature_names if f.startswith('feature_')]
    target_feature = numerical_features[0]
    
    # Gradual shift that creates different distribution shapes
    shift_values = np.linspace(0, 2, len(scenario_1))
    scenario_1[target_feature] += shift_values * scenario_1[target_feature].std()
    scenarios['gradual_mean_shift'] = scenario_1
    
    # Scenario 2: Distribution shape change (strong KL divergence signal)
    scenario_2 = test_df.copy()
    target_feature = numerical_features[1]
    
    # Transform from normal to exponential-like distribution
    original_data = scenario_2[target_feature]
    # Apply exponential transformation while preserving some original characteristics
    transformed_data = np.random.exponential(scale=np.abs(original_data.mean()), size=len(original_data))
    scenario_2[target_feature] = transformed_data
    scenarios['distribution_shape_change'] = scenario_2
    
    # Scenario 3: Categorical frequency drift (detectable by Chi-square and categorical KL)
    scenario_3 = test_df.copy()
    
    # Change category distribution significantly
    new_categories = np.random.choice(['Type_A', 'Type_B', 'Type_C'], 
                                    size=len(scenario_3), 
                                    p=[0.1, 0.2, 0.7])  # Very different from original [0.5, 0.3, 0.2]
    scenario_3['category_1'] = new_categories
    scenarios['categorical_frequency_drift'] = scenario_3
    
    # Scenario 4: Multiple subtle drifts (low individual signals, but cumulative effect)
    scenario_4 = test_df.copy()
    
    # Small shifts in multiple features
    for i, feature in enumerate(numerical_features[:4]):
        shift = 0.3 * scenario_4[feature].std() * (i + 1) / 4  # Increasing shifts
        scenario_4[feature] += shift
    
    # Slight categorical change
    mask = np.random.choice([True, False], size=len(scenario_4), p=[0.2, 0.8])
    scenario_4.loc[mask, 'category_2'] = 'Type_A'
    scenarios['multiple_subtle_drifts'] = scenario_4
    
    return scenarios

def compare_drift_detection_methods(reference_df, drifted_scenarios, y_ref, y_test, feature_names):
    """Compare standard XAdapt-Drift with advanced metrics."""
    
    # Train a model for impact analysis
    numerical_features = [f for f in feature_names if f.startswith('feature_')]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(reference_df[numerical_features], y_ref)
    
    # Create adapters and analyzers
    model_adapter = SklearnAdapter(model, feature_names=numerical_features)
    xadapt = XAdaptDrift(model_adapter=model_adapter, drift_threshold=0.05)
    advanced_detector = AdvancedDriftDetector(bins=50, alpha=0.05)
    
    results_comparison = {}
    
    for scenario_name, drifted_df in drifted_scenarios.items():
        print(f"\n{'='*60}")
        print(f"Analyzing Scenario: {scenario_name.upper()}")
        print(f"{'='*60}")
        
        # Standard XAdapt-Drift Analysis
        print("\n--- Standard XAdapt-Drift Analysis ---")
        xadapt_report = xadapt.analyze(
            reference=reference_df,
            current=drifted_df,
            y_reference=y_ref,
            y_current=y_test
        )
        
        print(f"Executive Summary: {xadapt_report['executive_summary']}")
        print(f"Drifted features (standard): {xadapt_report['drift_summary']['drifted_features']}")
        
        if 'impact_analysis' in xadapt_report:
            print(f"Global DIS: {xadapt_report['impact_analysis']['global_dis']:.2f}%")
        
        # Advanced Metrics Analysis
        print("\n--- Advanced Metrics Analysis ---")
        advanced_results = advanced_detector.comprehensive_drift_analysis(reference_df, drifted_df)
        advanced_summary = advanced_detector.create_drift_summary(advanced_results)
        
        print(f"Features with drift (advanced): {advanced_summary['drifted_features']}")
        print(f"Average KL divergence: {advanced_summary['average_kl_divergence']:.4f}")
        print(f"Average JS divergence: {advanced_summary['average_js_divergence']:.4f}")
        print(f"High confidence drifts: {advanced_summary['high_confidence_drifts']}")
        
        # Detailed comparison for key features
        print("\n--- Detailed Metrics Comparison ---")
        
        # Focus on features that show drift in either method
        all_drifted = set(xadapt_report['drift_summary']['drifted_features'] + 
                         advanced_summary['drifted_features'])
        
        for feature in sorted(all_drifted):
            if feature in advanced_results:
                adv_result = advanced_results[feature]
                print(f"\n{feature}:")
                print(f"  Feature type: {adv_result['feature_type']}")
                
                if adv_result['feature_type'] == 'numerical':
                    # Show standard vs advanced metrics
                    standard_detected = feature in xadapt_report['drift_summary']['drifted_features']
                    print(f"  Standard method detected drift: {standard_detected}")
                    
                    if feature in xadapt_report['drift_details']:
                        print(f"  Standard p-value: {xadapt_report['drift_details'][feature]['p_value']:.4f}")
                    
                    print(f"  KL divergence: {adv_result.get('kl_divergence', 'N/A'):.4f}")
                    print(f"  JS divergence: {adv_result.get('jensen_shannon_divergence', 'N/A'):.4f}")
                    print(f"  KS p-value: {adv_result.get('ks_pvalue', 'N/A'):.4f}")
                    print(f"  Wasserstein distance: {adv_result.get('wasserstein_distance', 'N/A'):.4f}")
                    print(f"  Mean shift: {adv_result.get('mean_shift', 'N/A'):.4f}")
                    
                else:  # categorical
                    standard_detected = feature in xadapt_report['drift_summary']['drifted_features']
                    print(f"  Standard method detected drift: {standard_detected}")
                    
                    if feature in xadapt_report['drift_details']:
                        print(f"  Standard p-value: {xadapt_report['drift_details'][feature]['p_value']:.4f}")
                    
                    print(f"  KL divergence: {adv_result.get('kl_divergence', 'N/A'):.4f}")
                    print(f"  JS divergence: {adv_result.get('jensen_shannon_divergence', 'N/A'):.4f}")
                    print(f"  Chi2 p-value: {adv_result.get('chi2_pvalue', 'N/A'):.4f}")
                    print(f"  New categories: {adv_result.get('new_categories', [])}")
                    print(f"  Missing categories: {adv_result.get('missing_categories', [])}")
                
                print(f"  Advanced method detected drift: {adv_result.get('overall_drift_detected', False)}")
                print(f"  Drift confidence: {adv_result.get('drift_confidence', 0):.2f}")
        
        # Store results for comparison
        results_comparison[scenario_name] = {
            'xadapt_summary': xadapt_report['drift_summary'],
            'advanced_summary': advanced_summary,
            'xadapt_drifted': set(xadapt_report['drift_summary']['drifted_features']),
            'advanced_drifted': set(advanced_summary['drifted_features'])
        }
    
    return results_comparison

def visualize_drift_metrics_comparison(reference_df, drifted_df, feature_name):
    """Visualize how different metrics capture the same drift."""
    
    advanced_detector = AdvancedDriftDetector(bins=30)
    
    # Calculate metrics
    ref_data = reference_df[feature_name].dropna()
    curr_data = drifted_df[feature_name].dropna()
    
    if pd.api.types.is_numeric_dtype(ref_data):
        kl_div = advanced_detector.kl_divergence(ref_data.values, curr_data.values)
        js_div = advanced_detector.jensen_shannon_divergence(ref_data.values, curr_data.values)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribution comparison
        axes[0, 0].hist(ref_data, bins=30, alpha=0.7, label='Reference', density=True)
        axes[0, 0].hist(curr_data, bins=30, alpha=0.7, label='Current', density=True)
        axes[0, 0].set_title(f'Distribution Comparison: {feature_name}')
        axes[0, 0].legend()
        axes[0, 0].set_ylabel('Density')
        
        # KDE plot
        axes[0, 1].set_title(f'KDE Comparison: {feature_name}')
        sns.kdeplot(ref_data, label='Reference', ax=axes[0, 1])
        sns.kdeplot(curr_data, label='Current', ax=axes[0, 1])
        axes[0, 1].legend()
        
        # Cumulative distribution
        axes[1, 0].set_title('Cumulative Distribution')
        ref_sorted = np.sort(ref_data)
        curr_sorted = np.sort(curr_data)
        ref_cdf = np.arange(1, len(ref_sorted) + 1) / len(ref_sorted)
        curr_cdf = np.arange(1, len(curr_sorted) + 1) / len(curr_sorted)
        
        axes[1, 0].plot(ref_sorted, ref_cdf, label='Reference')
        axes[1, 0].plot(curr_sorted, curr_cdf, label='Current')
        axes[1, 0].legend()
        axes[1, 0].set_ylabel('Cumulative Probability')
        
        # Metrics summary
        axes[1, 1].text(0.1, 0.8, f'KL Divergence: {kl_div:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f'JS Divergence: {js_div:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        
        # Add KS test result
        from scipy import stats
        ks_stat, ks_pvalue = stats.ks_2samp(ref_data, curr_data)
        axes[1, 1].text(0.1, 0.6, f'KS Statistic: {ks_stat:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f'KS p-value: {ks_pvalue:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        
        # Wasserstein distance
        wasserstein_dist = stats.wasserstein_distance(ref_data, curr_data)
        axes[1, 1].text(0.1, 0.4, f'Wasserstein Distance: {wasserstein_dist:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        
        axes[1, 1].set_title('Drift Metrics Summary')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig
    
    return None

def main():
    """Main function to run the comprehensive comparison."""
    
    print("Advanced Drift Metrics Integration Example")
    print("=" * 50)
    
    # Create datasets
    print("Creating synthetic datasets with various drift scenarios...")
    reference_df, test_df, y_ref, y_test, feature_names = create_enhanced_drift_scenarios()
    
    # Create drift scenarios
    print("Inducing different types of drift...")
    drifted_scenarios = induce_specific_drifts(test_df, feature_names)
    
    # Compare detection methods
    print("Comparing drift detection methods...")
    comparison_results = compare_drift_detection_methods(
        reference_df, drifted_scenarios, y_ref, y_test, feature_names
    )
    
    # Create summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON OF DETECTION METHODS")
    print("="*80)
    
    for scenario_name, results in comparison_results.items():
        print(f"\nScenario: {scenario_name}")
        xadapt_count = len(results['xadapt_drifted'])
        advanced_count = len(results['advanced_drifted'])
        
        print(f"  Standard XAdapt-Drift detected: {xadapt_count} features")
        print(f"  Advanced metrics detected: {advanced_count} features")
        
        # Find agreement and disagreement
        agreement = results['xadapt_drifted'] & results['advanced_drifted']
        only_xadapt = results['xadapt_drifted'] - results['advanced_drifted']
        only_advanced = results['advanced_drifted'] - results['xadapt_drifted']
        
        print(f"  Agreement on: {list(agreement)}")
        if only_xadapt:
            print(f"  Only XAdapt-Drift detected: {list(only_xadapt)}")
        if only_advanced:
            print(f"  Only advanced metrics detected: {list(only_advanced)}")
        
        print(f"  Average KL divergence: {results['advanced_summary']['average_kl_divergence']:.4f}")
        print(f"  Average JS divergence: {results['advanced_summary']['average_js_divergence']:.4f}")
    
    # Visualize one example
    scenario_to_visualize = 'distribution_shape_change'
    if scenario_to_visualize in drifted_scenarios:
        print(f"\nCreating visualization for {scenario_to_visualize} scenario...")
        numerical_features = [f for f in feature_names if f.startswith('feature_')]
        target_feature = numerical_features[1]  # The one we modified
        
        fig = visualize_drift_metrics_comparison(
            reference_df, drifted_scenarios[scenario_to_visualize], target_feature
        )
        
        if fig:
            plt.savefig(f'drift_metrics_comparison_{target_feature}.png', dpi=300, bbox_inches='tight')
            print(f"Visualization saved as 'drift_metrics_comparison_{target_feature}.png'")
    
    print("\nAnalysis complete! The advanced metrics provide additional insights into the nature and magnitude of drift.")

if __name__ == "__main__":
    main()
