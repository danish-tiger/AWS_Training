import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from housing_predictor.models.scoring import prepare_test_features, score_model


@pytest.fixture
def sample_test_data():
    return pd.DataFrame(
        {
            "longitude": [-122.23, -122.22],
            "latitude": [37.88, 37.86],
            "housing_median_age": [41, 21],
            "total_rooms": [880, 7099],
            "total_bedrooms": [129, 1106],
            "population": [322, 2401],
            "households": [126, 1138],
            "median_income": [8.3252, 8.3014],
            "median_house_value": [452600, 358500],
            "ocean_proximity": ["NEAR BAY", "NEAR BAY"],
        }
    )


@pytest.fixture
def dummy_model():
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    # Dummy training data
    X = np.random.rand(10, 5)
    y = np.random.rand(10)
    model.fit(X, y)
    return model


@pytest.fixture
def dummy_imputer(sample_test_data):
    imputer = SimpleImputer(strategy="median")
    # Fit with the same columns as our test data
    test_num = sample_test_data.drop(["median_house_value", "ocean_proximity"], axis=1)
    imputer.fit(test_num)
    return imputer


def test_prepare_test_features(sample_test_data, dummy_imputer):
    features, labels = prepare_test_features(sample_test_data, dummy_imputer)

    assert isinstance(features, pd.DataFrame)
    assert isinstance(labels, pd.Series)
    assert len(features) == len(labels) == len(sample_test_data)
    assert "ocean_proximity" not in features.columns
    assert "median_house_value" not in features.columns
    assert "rooms_per_household" in features.columns


def test_score_model(sample_test_data, dummy_imputer):
    # Prepare test features (this will create 11 features)
    test_features, test_labels = prepare_test_features(sample_test_data, dummy_imputer)

    # Create a dummy model with the correct number of features
    n_features = test_features.shape[1]
    model = RandomForestRegressor(n_estimators=10, random_state=42)

    # Train with dummy data that matches the feature dimensions
    X_dummy = np.random.rand(10, n_features)
    y_dummy = np.random.rand(10)
    model.fit(X_dummy, y_dummy)

    # Now scoring should work
    scores = score_model(model, test_features, test_labels)

    assert isinstance(scores, dict)
    assert "rmse" in scores
    assert "mae" in scores
    assert "predictions" in scores
    assert len(scores["predictions"]) == len(test_labels)
