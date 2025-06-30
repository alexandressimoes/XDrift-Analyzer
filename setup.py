from setuptools import setup, find_packages

setup(
    name="xadapt-drift",
    version="0.0.1",
    description="A simple drift detection package",
    author="Alexandre Simões",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "evidently",
        "scikit-learn",
        "scipy",
        "nannyml"
    ],
    license="MIT",
)