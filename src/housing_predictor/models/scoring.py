import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


def prepare_test_features(test_data, imputer):
    """Prepare test data features for scoring."""
    logger.info("Preparing test data features")

    # Separate features and labels
    test_labels = test_data["median_house_value"].copy()
    test_data = test_data.drop("median_house_value", axis=1)

    # Handle numerical features
    test_num = test_data.drop("ocean_proximity", axis=1)
    test_num_imputed = imputer.transform(test_num)

    test_prepared = pd.DataFrame(
        test_num_imputed, columns=test_num.columns, index=test_data.index
    )

    # Add engineered features
    test_prepared["rooms_per_household"] = (
        test_prepared["total_rooms"] / test_prepared["households"]
    )
    test_prepared["bedrooms_per_room"] = (
        test_prepared["total_bedrooms"] / test_prepared["total_rooms"]
    )
    test_prepared["population_per_household"] = (
        test_prepared["population"] / test_prepared["households"]
    )

    # Handle categorical features
    test_cat = test_data[["ocean_proximity"]]
    test_prepared = test_prepared.join(pd.get_dummies(test_cat, drop_first=True))

    return test_prepared, test_labels


def score_model(model, test_prepared, test_labels):
    """Score the model on test data."""
    logger.info("Scoring model on test data")

    predictions = model.predict(test_prepared)
    mse = mean_squared_error(test_labels, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_labels, predictions)

    logger.info("Test RMSE: %.2f", rmse)
    logger.info("Test MAE: %.2f", mae)

    return {"rmse": rmse, "mae": mae, "predictions": predictions}
