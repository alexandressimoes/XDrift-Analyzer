"""
Visualization utilities for XAdapt-Drift.

This module provides visualization functions to help interpret drift analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union

def plot_numerical_drift(reference_series: pd.Series, 
                        current_series: pd.Series,
                        title: Optional[str] = None,
                        figsize: Tuple[int, int] = (10, 6)):
    """
    Plot the distribution of a numerical feature to visualize drift.
    
    Args:
        reference_series: Reference data series
        current_series: Current data series
        title: Plot title
        figsize: Figure size as (width, height)
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    plt.figure(figsize=figsize)
    
    sns.kdeplot(reference_series, label='Reference', fill=True, alpha=0.3)
    sns.kdeplot(current_series, label='Current', fill=True, alpha=0.3)
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Distribution Drift: {reference_series.name}')
        
    plt.xlabel(reference_series.name)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_categorical_drift(reference_series: pd.Series,
                          current_series: pd.Series,
                          title: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 6)):
    """
    Plot the distribution of a categorical feature to visualize drift.
    
    Args:
        reference_series: Reference data series
        current_series: Current data series
        title: Plot title
        figsize: Figure size as (width, height)
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    plt.figure(figsize=figsize)
    
    ref_counts = reference_series.value_counts(normalize=True)
    curr_counts = current_series.value_counts(normalize=True)
    
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
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Frequency Drift: {reference_series.name}')
        
    plt.xticks(x, all_cats, rotation=45 if len(all_cats) > 5 else 0)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

def plot_drift_impact(impact_results: Dict[str, Any], 
                     max_features: int = 10,
                     figsize: Tuple[int, int] = (12, 8)):
    """
    Plot the Drift Impact Score (DIS) for top features.
    
    Args:
        impact_results: The impact_analysis section from XAdaptDrift report
        max_features: Maximum number of features to display
        figsize: Figure size as (width, height)
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    plt.figure(figsize=figsize)
    
    feature_impacts = []
    
    for feature, impact in impact_results['feature_impact'].items():
        feature_impacts.append({
            'feature': feature,
            'drift_impact_score': impact['drift_impact_score'],
            'abs_impact': abs(impact['drift_impact_score']),
            'had_drift': impact['had_drift'],
            'reference_importance': impact['reference_importance']
        })
    
    impact_df = pd.DataFrame(feature_impacts)
    
    # Sort by absolute impact score
    impact_df = impact_df.sort_values('abs_impact', ascending=False)
    
    # Take top N features
    impact_df = impact_df.head(max_features)
    
    # Sort by impact score (not absolute) for display
    impact_df = impact_df.sort_values('drift_impact_score')
    
    # Create color mapping
    colors = ['red' if row['had_drift'] else 'gray' for _, row in impact_df.iterrows()]
    
    bars = plt.barh(impact_df['feature'], impact_df['drift_impact_score'], color=colors)
    
    # Add a line at zero
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 5 if width > 0 else width - 5
        ha = 'left' if width > 0 else 'right'
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                va='center', ha=ha, fontsize=9)
    
    plt.xlabel('Drift Impact Score (%)')
    plt.ylabel('Feature')
    plt.title('Drift Impact on Feature Importance')
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Drifted Feature'),
        Patch(facecolor='gray', label='Non-Drifted Feature')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_global_dis_history(global_dis_values: List[float], 
                           timestamps: Optional[List[str]] = None,
                           threshold: Optional[float] = None,
                           figsize: Tuple[int, int] = (12, 6)):
    """
    Plot the Global DIS over time.
    
    Args:
        global_dis_values: List of Global DIS values
        timestamps: List of timestamps for each value
        threshold: Optional threshold to highlight critical values
        figsize: Figure size as (width, height)
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    plt.figure(figsize=figsize)
    
    x = timestamps if timestamps else list(range(len(global_dis_values)))
    
    plt.plot(x, global_dis_values, 'o-', linewidth=2, markersize=8)
    
    # Add threshold line if provided
    if threshold is not None:
        plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, 
                   label=f'Threshold ({threshold}%)')
        plt.legend()
    
    plt.xlabel('Time' if timestamps else 'Sample')
    plt.ylabel('Global Drift Impact Score (%)')
    plt.title('Global DIS Over Time')
    plt.grid(True, alpha=0.3)
    
    if timestamps:
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
    
    plt.tight_layout()
    
    return plt.gcf()

def create_drift_dashboard(report: Dict[str, Any], 
                          reference_df: pd.DataFrame,
                          current_df: pd.DataFrame,
                          output_file: Optional[str] = None):
    """
    Create a comprehensive drift analysis dashboard.
    
    Args:
        report: XAdaptDrift analysis report
        reference_df: Reference dataframe
        current_df: Current dataframe
        output_file: File path to save the dashboard (e.g., 'dashboard.png')
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Get drifted features
    drifted_features = report['drift_summary'].get('drifted_features', [])
    
    if not drifted_features:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No drift detected", 
               horizontalalignment='center', verticalalignment='center',
               fontsize=16)
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
        return fig
    
    # Count number of plots needed
    n_drift_plots = min(len(drifted_features), 3)  # Show at most 3 drifted features
    has_impact = 'impact_analysis' in report
    
    if has_impact:
        n_rows = 2
        n_cols = max(n_drift_plots, 1)
    else:
        n_rows = 1
        n_cols = n_drift_plots
    
    fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
    
    # Plot drifted features
    for i, feature in enumerate(drifted_features[:n_drift_plots]):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        feature_type = report['drift_details'][feature]['feature_type']
        
        if feature_type == 'numerical':
            sns.kdeplot(reference_df[feature], label='Reference', fill=True, alpha=0.3, ax=ax)
            sns.kdeplot(current_df[feature], label='Current', fill=True, alpha=0.3, ax=ax)
            ax.set_title(f'Drift in {feature}')
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ref_counts = reference_df[feature].value_counts(normalize=True)
            curr_counts = current_df[feature].value_counts(normalize=True)
            all_cats = sorted(set(ref_counts.index) | set(curr_counts.index))
            
            x = np.arange(len(all_cats))
            width = 0.35
            
            ref_values = [ref_counts.get(cat, 0) for cat in all_cats]
            curr_values = [curr_counts.get(cat, 0) for cat in all_cats]
            
            ax.bar(x - width/2, ref_values, width, label='Reference')
            ax.bar(x + width/2, curr_values, width, label='Current')
            ax.set_title(f'Drift in {feature}')
            ax.set_xlabel('Category')
            ax.set_ylabel('Frequency')
            ax.set_xticks(x)
            ax.set_xticklabels(all_cats, rotation=45 if len(all_cats) > 5 else 0)
            ax.legend()
    
    # Plot impact analysis
    if has_impact:
        ax = fig.add_subplot(n_rows, 1, 2)
        
        feature_impacts = []
        for feature, impact in report['impact_analysis']['feature_impact'].items():
            if impact['had_drift']:
                feature_impacts.append({
                    'feature': feature,
                    'drift_impact_score': impact['drift_impact_score'],
                    'abs_impact': abs(impact['drift_impact_score'])
                })
        
        if feature_impacts:
            impact_df = pd.DataFrame(feature_impacts)
            impact_df = impact_df.sort_values('drift_impact_score')
            
            bars = ax.barh(impact_df['feature'], impact_df['drift_impact_score'], 
                          color=['red' if val < 0 else 'blue' for val in impact_df['drift_impact_score']])
                          
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + 5 if width > 0 else width - 5
                ha = 'left' if width > 0 else 'right'
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                       va='center', ha=ha, fontsize=9)
            
            ax.set_xlabel('Drift Impact Score (%)')
            ax.set_ylabel('Feature')
            ax.set_title(f'Drift Impact on Features (Global DIS: {report["impact_analysis"]["global_dis"]:.1f}%)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No significant impact found", 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=16)
    
    # Add the executive summary as a text box
    if 'executive_summary' in report:
        plt.figtext(0.5, 0.01, report['executive_summary'],
                   wrap=True, horizontalalignment='center',
                   fontsize=12, bbox=dict(boxstyle="round", 
                                         fc="0.9", 
                                         ec="0.5", 
                                         alpha=0.9))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the text box
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
    
    return fig
