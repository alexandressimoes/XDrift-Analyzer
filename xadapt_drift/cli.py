"""
Command-line interface for XAdapt-Drift.

This module provides a CLI for running XAdapt-Drift analyses.
"""

import argparse
import pandas as pd
import numpy as np
import json
import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
import joblib

from xadapt_drift import XAdaptDrift
from xadapt_drift.adapters.sklearn_adapter import SklearnAdapter
from xadapt_drift.utils.visualization import create_drift_dashboard

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='XAdapt-Drift: Data drift analysis and explainability'
    )
    
    parser.add_argument('--reference', type=str, required=True,
                        help='Path to reference data CSV file')
    parser.add_argument('--current', type=str, required=True,
                        help='Path to current data CSV file')
    parser.add_argument('--model', type=str, required=False,
                        help='Path to saved model file (e.g., .pkl, .joblib)')
    parser.add_argument('--y-reference', type=str, required=False,
                        help='Path to reference target CSV file')
    parser.add_argument('--y-current', type=str, required=False,
                        help='Path to current target CSV file')
    parser.add_argument('--output', type=str, default='drift_report.json',
                        help='Path to output report file')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='p-value threshold for drift detection')
    parser.add_argument('--method', type=str, choices=['shap', 'permutation'], 
                        default='permutation',
                        help='Method for feature importance')
    parser.add_argument('--dashboard', type=str, default=None,
                        help='Path to save visual dashboard (e.g., dashboard.png)')
    
    return parser.parse_args()

def load_data(file_path):
    """Load data from CSV or other formats."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext == '.parquet':
        return pd.read_parquet(file_path)
    elif ext == '.pkl' or ext == '.pickle':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def load_model(model_path):
    """Load a machine learning model."""
    ext = os.path.splitext(model_path)[1].lower()
    
    try:
        if ext == '.pkl' or ext == '.pickle':
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        elif ext == '.joblib':
            return joblib.load(model_path)
        else:
            raise ValueError(f"Unsupported model format: {ext}")
    except Exception as e:
        raise ValueError(f"Error loading model: {str(e)}")

def main():
    """Run XAdapt-Drift analysis from command-line arguments."""
    args = parse_args()
    
    print("XAdapt-Drift Analysis")
    print("=====================")
    
    # Load data
    print(f"Loading reference data from {args.reference}...")
    reference_df = load_data(args.reference)
    
    print(f"Loading current data from {args.current}...")
    current_df = load_data(args.current)
    
    # Load target values if provided
    y_reference = None
    y_current = None
    
    if args.y_reference:
        print(f"Loading reference target from {args.y_reference}...")
        y_reference_data = load_data(args.y_reference)
        if isinstance(y_reference_data, pd.DataFrame):
            y_reference = y_reference_data.values.ravel()
        else:
            y_reference = y_reference_data
            
    if args.y_current:
        print(f"Loading current target from {args.y_current}...")
        y_current_data = load_data(args.y_current)
        if isinstance(y_current_data, pd.DataFrame):
            y_current = y_current_data.values.ravel()
        else:
            y_current = y_current_data
    
    # Create XAdapt-Drift analyzer
    model_adapter = None
    if args.model:
        print(f"Loading model from {args.model}...")
        model = load_model(args.model)
        print("Creating model adapter...")
        model_adapter = SklearnAdapter(model, feature_names=reference_df.columns.tolist())
    
    # Create analyzer
    xadapt = XAdaptDrift(
        model_adapter=model_adapter,
        drift_threshold=args.threshold,
        explanation_method=args.method
    )
    
    # Run analysis
    print("Running drift analysis...")
    report = xadapt.analyze(
        reference=reference_df,
        current=current_df,
        y_reference=y_reference,
        y_current=y_current
    )
    
    # Save report
    print(f"Saving report to {args.output}...")
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n=== Analysis Summary ===")
    print(report["executive_summary"])
    
    if "drift_summary" in report:
        n_drift = report["drift_summary"]["drifted_features_count"]
        n_total = report["drift_summary"]["total_features_analyzed"]
        print(f"\nDrift detected in {n_drift} out of {n_total} features "
              f"({n_drift/n_total:.1%})")
    
    if "impact_analysis" in report:
        print(f"Global Drift Impact Score: {report['impact_analysis']['global_dis']:.2f}%")
    
    # Create dashboard if requested
    if args.dashboard and model_adapter:
        print(f"Creating and saving dashboard to {args.dashboard}...")
        create_drift_dashboard(report, reference_df, current_df, args.dashboard)
        print(f"Dashboard saved to {args.dashboard}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
