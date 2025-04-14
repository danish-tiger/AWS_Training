import argparse
import os
import pickle

import pandas as pd
from sklearn.impute import SimpleImputer

from housing_predictor.data.ingestion import prepare_data
from housing_predictor.models.scoring import prepare_test_features, score_model
from housing_predictor.utils.logging_config import configure_logging


def main():
    parser = argparse.ArgumentParser(description="Score housing price prediction model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model pickle file",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to test data CSV file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results",
        help="Path to save scoring results",
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

    # Load model
    with open(args.model_path, "rb") as f:
        model = pickle.load(f)

    # Load and prepare test data
    test_data = pd.read_csv(args.data_path)
    test_data_prepared = prepare_data(test_data)

    # Create imputer (same as used in training)
    housing_num = test_data_prepared.drop(
        ["median_house_value", "ocean_proximity"], axis=1, errors="ignore"
    )
    imputer = SimpleImputer(strategy="median")
    imputer.fit(housing_num)

    # Prepare features and score
    test_features, test_labels = prepare_test_features(test_data_prepared, imputer)
    scores = score_model(model, test_features, test_labels)

    # Save results
    os.makedirs(args.output_path, exist_ok=True)
    results_path = os.path.join(args.output_path, "scoring_results.txt")
    with open(results_path, "w") as f:
        f.write(f"RMSE: {scores['rmse']}\n")
        f.write(f"MAE: {scores['mae']}\n")

    print(f"Results saved to {results_path}")
    print(f"RMSE: {scores['rmse']}")
    print(f"MAE: {scores['mae']}")


if __name__ == "__main__":
    main()
