# setup.py
from setuptools import setup, find_packages

setup(
    name="multi-echelon-ppo",
    version="0.1.0",
    description="Multi-Echelon Inventory Optimization with PPO",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "gymnasium>=0.28.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tensorboard>=2.6.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0",
    ],
    python_requires=">=3.8",
)