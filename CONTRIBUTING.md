# Contributing to XAdapt-Drift

Thank you for your interest in contributing to XAdapt-Drift! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and considerate of others. We aim to foster an open and welcoming environment for everyone.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with the following information:

- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Any error messages or logs
- Your environment (Python version, OS, package versions)

### Suggesting Features

To suggest a feature:

1. Check if the feature has already been suggested
2. Create a new issue with the "enhancement" label
3. Describe the feature, why it's useful, and how it should work

### Pull Requests

1. Fork the repository
2. Create a new branch for your changes
3. Make your changes
4. Add or update tests for your changes
5. Ensure all tests pass
6. Update documentation if needed
7. Submit a pull request

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/yourusername/xadapt-drift.git
   cd xadapt-drift
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Run tests:
   ```bash
   pytest
   ```

## Code Style

We follow PEP 8 style guidelines. Please format your code with `black` and ensure it passes `flake8` checks:

```bash
black xadapt_drift
flake8 xadapt_drift
```

## Documentation

- Document all public modules, classes, and functions
- Follow the NumPy docstring style
- Update the README.md with any necessary changes

## Architecture

XAdapt-Drift follows a modular architecture:

1. **ModelAdapter**: Abstracts interactions with ML libraries
2. **DriftDetector**: Detects statistical drift
3. **DriftCharacterizer**: Analyzes drift characteristics
4. **ImpactAnalyzer**: Quantifies impact on model behavior
5. **ReportGenerator**: Produces reports

When extending XAdapt-Drift, please ensure your changes fit within this architecture.

## Adding New Adapters

To add support for a new ML library:

1. Create a new file in `xadapt_drift/adapters/`
2. Implement the `BaseAdapter` interface
3. Add tests in `tests/adapters/`
4. Update documentation

## Adding New Drift Detection Methods

To add new drift detection methods:

1. Extend the `DriftDetector` class in `xadapt_drift/drift/detector.py`
2. Add tests in `tests/drift/`
3. Update documentation

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.
