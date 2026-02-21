from setuptools import setup, find_packages

setup(
    name="tf-litho",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.12.0",
        "numpy>=1.19.0", 
        "scipy>=1.7.0",
        "scikit-image>=0.19.0",
        "matplotlib>=3.3.0"
    ],
    python_requires=">=3.8",
)