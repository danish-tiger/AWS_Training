import os
import tarfile

import pandas as pd
import pytest

from housing_predictor.data.ingestion import (
    fetch_housing_data,
    load_housing_data,
    split_data,
)


@pytest.fixture
def sample_housing_data():
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


# def test_fetch_housing_data(tmp_path):
#     test_path = tmp_path / "data" / "housing"
#     fetch_housing_data(housing_path=str(test_path))
#     assert os.path.exists(str(test_path / "housing.csv"))
def test_fetch_housing_data(tmp_path):
    test_path = tmp_path / "data" / "housing"
    test_path.mkdir(parents=True)

    # Create a dummy tar.gz file to simulate fallback
    tgz_path = test_path / "housing.tgz"
    with tarfile.open(tgz_path, "w:gz") as tar:
        dummy_file = test_path / "dummy.txt"
        dummy_file.write_text("test")
        tar.add(dummy_file, arcname="dummy.txt")

    # Now call your function with this test path
    fetch_housing_data(housing_path=str(test_path))

    assert (test_path / "housing.tgz").exists()


def test_load_housing_data(tmp_path):
    test_path = tmp_path / "data" / "housing"
    os.makedirs(test_path, exist_ok=True)

    # Create a dummy CSV file
    dummy_data = "longitude,latitude\n-122.23,37.88\n-122.22,37.86"
    with open(test_path / "housing.csv", "w") as f:
        f.write(dummy_data)

    data = load_housing_data(housing_path=str(test_path))
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 2


def test_split_data(sample_housing_data):
    train_set, test_set = split_data(sample_housing_data)
    assert len(train_set) + len(test_set) == len(sample_housing_data)
    assert isinstance(train_set, pd.DataFrame)
    assert isinstance(test_set, pd.DataFrame)
