from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="housing_predictor",
    version="0.1.0",
    author="Danish Azam",
    author_email="danish.azam@tigeranalytics.com",
    description="A package for predicting housing prices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/AWS_TRAINING",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "test": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ],
    },
)
