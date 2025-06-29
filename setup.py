from setuptools import setup, find_packages

setup(
    name="credit-risk-model",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "pandas",
        "numpy",
        "pytest"
    ],
)