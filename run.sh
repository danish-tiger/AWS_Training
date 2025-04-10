#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

echo "🔍 Running code style checks..."

echo "Running black..."
black .
black --check .

echo "Running isort..."
isort .
isort --check .

echo "Running flake8..."
flake8 .

echo "✅ Style checks passed."

echo "🧪 Running tests with coverage..."
pytest -v --cov=housing_predictor

echo "✅ Tests passed."

echo "🚀 Running scripts with logging..."

echo "Running ingest_data.py..."
python scripts/ingest_data.py --output-path data/processed --log-level DEBUG

echo "Running train.py..."
python scripts/train.py --input-path data/processed/housing_train.csv --output-path models --log-level DEBUG

echo "Running score.py..."
python scripts/score.py --model-path models/housing_model.pkl --data-path data/processed/housing_test.csv --output-path results --log-level DEBUG

echo "✅ All steps completed successfully."
