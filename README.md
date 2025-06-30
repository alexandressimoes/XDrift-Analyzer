# XAdapt-Drift

A Python-based open-source framework for data drift analysis and explainability. XAdapt-Drift goes beyond simple drift detection to provide comprehensive diagnosis and impact analysis of data drift in machine learning models.

## Features

- **Advanced Drift Detection**: Detect data drift across numerical and categorical features
- **Drift Characterization**: Understand why drift occurred with detailed statistical analysis
- **Impact Analysis**: Quantify how drift affects model behavior using feature importance
- **Drift Impact Score (DIS)**: Measure the impact of drift on each feature's importance
- **Global DIS**: A single KPI to track model health in relation to drift
- **Flexible Integration**: Works with existing models and different ML libraries
- **Rich Reports**: Generate comprehensive reports with actionable insights

## Installation

```bash
# From PyPI (coming soon)
pip install xadapt-drift

# From source
git clone https://github.com/alexandress/xadapt-drift.git
cd xadapt-drift
pip install -e .
```

## Quick Start

```python
import pandas as pd
from xadapt_drift import XAdaptDrift
from xadapt_drift.adapters.sklearn_adapter import SklearnAdapter

# Load your model and data
model_adapter = SklearnAdapter(your_model)
xadapt = XAdaptDrift(model_adapter=model_adapter)

# Analyze drift between reference and current data
report = xadapt.analyze(
    reference=reference_df,  # Your reference/training data
    current=current_df,      # Your current/production data
    y_reference=y_reference  # Optional, for permutation importance
)

# View the results
print(report["executive_summary"])
print(f"Global DIS: {report['impact_analysis']['global_dis']:.2f}%")
```

## Why XAdapt-Drift?

Traditional drift detection tools only tell you **if** drift occurred. XAdapt-Drift answers:

1. **Why did drift occur?** Through detailed characterization of statistical changes
2. **So what?** By measuring the actual impact on your model's behavior

This allows MLOps teams to prioritize actions based on business impact rather than statistical significance alone.

## Architecture

XAdapt-Drift follows a modular, extensible design:

1. **ModelAdapter**: Standardizes interaction with ML libraries (scikit-learn, XGBoost, PyTorch)
2. **DriftDetector**: Identifies statistical drift in features
3. **DriftCharacterizer**: Analyzes the nature of detected drift
4. **ImpactAnalyzer**: Quantifies how drift affects model behavior
5. **ReportGenerator**: Produces comprehensive, actionable reports

## Contributing

Contributions are welcome! Please check our contribution guidelines for more details.

## License

MIT License
