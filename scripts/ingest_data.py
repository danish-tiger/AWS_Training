import argparse
import os

from housing_predictor.data.ingestion import (
    fetch_housing_data,
    load_housing_data,
    split_data,
)
from housing_predictor.utils.logging_config import configure_logging


def main():
    parser = argparse.ArgumentParser(description="Fetch and prepare housing data")
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/processed",
        help="Path to save processed data",
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

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Fetch and process data
    fetch_housing_data()
    housing = load_housing_data()
    train_set, test_set = split_data(housing)

    # Save processed data
    train_path = os.path.join(args.output_path, "housing_train.csv")
    test_path = os.path.join(args.output_path, "housing_test.csv")

    train_set.to_csv(train_path, index=False)
    test_set.to_csv(test_path, index=False)

    print(f"Training data saved to {train_path}")
    print(f"Test data saved to {test_path}")


if __name__ == "__main__":
    main()
