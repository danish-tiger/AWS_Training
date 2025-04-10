import pytest
import pandas as pd
import numpy as np
from housing_predictor.models.training import prepare_features, train_model

@pytest.fixture
def sample_housing_data():
    return pd.DataFrame({
        "longitude": [-122.23, -122.22, -122.24],
        "latitude": [37.88, 37.86, 37.85],
        "housing_median_age": [41, 21, 52],
        "total_rooms": [880, 7099, 1467],
        "total_bedrooms": [129, 1106, 190],
        "population": [322, 2401, 496],
        "households": [126, 1138, 177],
        "median_income": [8.3252, 8.3014, 7.2574],
        "median_house_value": [452600, 358500, 352100],
        "ocean_proximity": ["NEAR BAY", "NEAR BAY", "INLAND"],
    })

def test_prepare_features(sample_housing_data):
    features, labels = prepare_features(sample_housing_data)

    assert isinstance(features, pd.DataFrame)
    assert isinstance(labels, pd.Series)
    assert len(features) == len(labels) == len(sample_housing_data)
    assert "ocean_proximity" not in features.columns
    assert "median_house_value" not in features.columns
    assert "rooms_per_household" in features.columns

def test_train_model(sample_housing_data):
    features, labels = prepare_features(sample_housing_data)
    model = train_model(features, labels)

    assert hasattr(model, "predict")
    predictions = model.predict(features)
    assert len(predictions) == len(labels)
    assert isinstance(predictions, np.ndarray)