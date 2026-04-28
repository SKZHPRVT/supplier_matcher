"""Setup configuration for supplier_matcher package."""
from setuptools import setup, find_packages

setup(
    name="supplier_matcher",
    version="0.1.0",
    description="AI-powered supplier matching system for procurement optimization",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "openpyxl>=3.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
    ],
    python_requires=">=3.9",
)