"""
Setup script for Quant AI Trading System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = []

# Development requirements
dev_requirements = [
    'pytest>=5.4.1',
    'pytest-cov>=2.10.0',
    'black>=20.8b1',
    'flake8>=3.8.3',
    'mypy>=0.782',
    'jupyter>=1.0.0',
    'ipython>=7.13.0',
]

setup(
    name="quant-ai-trading",
    version="1.0.0",
    author="Quant AI Trading Team",
    author_email="",
    description="A comprehensive quantitative trading system combining traditional quant strategies, ML, and DL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/m1nd0322/algoTrade",
    packages=find_packages(exclude=['tests', 'notebooks', 'scripts']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    extras_require={
        'dev': dev_requirements,
    },
    entry_points={
        'console_scripts': [
            'quant-collect=scripts.run_data_collection:main',
            'quant-preprocess=scripts.run_preprocessing:main',
            'quant-backtest=scripts.run_backtesting:main',
            'quant-analyze=scripts.run_analysis:main',
            'quant-pipeline=scripts.run_full_pipeline:main',
        ],
    },
    include_package_data=True,
    package_data={
        'src': [
            'reporting/templates/*.md',
            'reporting/templates/*.html',
        ],
    },
    zip_safe=False,
    keywords=[
        'trading',
        'quantitative',
        'finance',
        'algorithmic-trading',
        'machine-learning',
        'deep-learning',
        'backtesting',
        'investment',
        'stock-market',
    ],
    project_urls={
        'Bug Reports': 'https://github.com/m1nd0322/algoTrade/issues',
        'Source': 'https://github.com/m1nd0322/algoTrade',
    },
)
