import argparse
import os
import pickle

import pandas as pd

from housing_predictor.data.ingestion import prepare_data
from housing_predictor.models.training import prepare_features, train_model
from housing_predictor.utils.logging_config import configure_logging


def main():
    parser = argparse.ArgumentParser(description="Train housing price prediction model")
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to training data CSV file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="models",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Path to log file (default: no file logging)",
    )
    parser.add_argument(
        "--no-console-log",
        action="store_false",
        dest="console_log",
        help="Disable console logging",
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging(
        log_level=args.log_level,
        log_path=args.log_path,
        console_log=args.console_log,
    )

    # Load data
    housing = pd.read_csv(args.input_path)

    # Prepare data
    housing_prepared = prepare_data(housing)
    features, labels = prepare_features(housing_prepared)

    # Train model
    model = train_model(features, labels)

    # Save model
    os.makedirs(args.output_path, exist_ok=True)
    model_path = os.path.join(args.output_path, "housing_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
