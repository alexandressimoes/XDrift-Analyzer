from setuptools import setup, find_packages

setup(
    name="xadapt-drift",
    version="0.1.0",
    description="Python framework for data drift analysis and explainability",
    author="Alexandre Simões",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "shap>=0.40.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0"
    ],
    entry_points={
        'console_scripts': [
            'xadapt-drift=xadapt_drift.cli:main',
        ],
    },
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "docs": ["sphinx", "sphinx_rtd_theme"]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    license="MIT",
)