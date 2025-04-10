import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import logging

logger = logging.getLogger(__name__)


def prepare_features(housing):
    """Prepare features for training."""
    logger.info("Preparing features for training")

    # Separate features and labels
    housing_labels = housing["median_house_value"].copy()
    housing = housing.drop("median_house_value", axis=1)

    # Handle numerical features
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer = SimpleImputer(strategy="median")
    housing_num_imputed = imputer.fit_transform(housing_num)

    housing_tr = pd.DataFrame(
        housing_num_imputed, columns=housing_num.columns, index=housing.index
    )

    # Add engineered features
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    # Handle categorical features
    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    return housing_prepared, housing_labels


def train_model(housing_prepared, housing_labels):
    """Train a Random Forest model with grid search."""
    logger.info("Training Random Forest model")

    param_grid = [
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)

    logger.info("Best parameters: %s", grid_search.best_params_)

    return grid_search.best_estimator_