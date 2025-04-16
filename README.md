# Housing Price Prediction Package

A Python package for predicting housing prices using machine learning.

## Features

- Data ingestion and preprocessing
- Model training with Random Forest
- Model evaluation and scoring
- Logging and configuration
- CI/CD pipeline with GitHub Actions

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AWS_TRAINING.git
   cd AWS_TRAINING


2. Create and activate the conda environment:

conda env create -f env.yml
conda activate housing_predictor

3. Install the package in development mode:

pip install -e .


Usage
1. Ingest data:
python scripts/ingest_data.py --output-path data/processed
2. Train model:
python scripts/train.py --input-path data/processed/housing_train.csv --output-path models
3. Score model:
python scripts/score.py --model-path models/housing_model.pkl --data-path data/processed/housing_test.csv --output-path results

Usage of docker to test locally
# From project root (where Dockerfile is)
docker build -t housing-predictor .

# Test the main pipeline
docker run --rm housing-predictor