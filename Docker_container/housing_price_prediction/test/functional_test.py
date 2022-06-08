"""Functional test for the whole housing_packaged.
"""
import os
from glob import glob

import pytest


def test_ingest():
    """Tests ingest_data.py module."""
    raw = "data/raw/"
    processed = "data/processed/"
    os.system(f"python housing_packaged/ingest_data.py --raw {raw} --processed {processed}")
    assert os.path.isfile(f"{raw}/housing.csv")
    assert os.path.isfile(f"{processed}/housing_train.csv")
    assert os.path.isfile(f"{processed}/housing_test.csv")


def test_train():
    """Tests train.py module."""
    models = "data/models/"
    dataset = "data/processed/housing_train.csv"
    os.system(f"python housing_packaged/train.py -d {dataset} -m {models}")
    assert os.path.isfile(f"{models}/LinearRegression.pkl")
    assert os.path.isfile(f"{models}/RandomForestRegressor.pkl")
    assert os.path.isfile(f"{models}/DecisionTreeRegressor.pkl")


def test_score(cleanup):
    """Tests score.py module."""
    models = "data/models"
    dataset = "data/processed/housing_test.csv"
    log_file = "log_file.txt"

    os.system(f"python housing_packaged/score.py -d {dataset} -m {models} --mae --rmse --log-path {log_file}")

    with open(log_file, "r") as f:
        lines = f.readlines()

    assert len(lines) == 17
    assert lines[0].startswith("Fetched")
    assert lines[1].startswith("Preprocessing")
    assert lines[2].startswith("Preprocessing")
    assert lines[3].startswith("Saving")
    assert lines[4].startswith("Preprocessed train")
    assert lines[5].startswith("Preprocessed test")
    assert lines[6].startswith("Started training.")
    assert lines[7].startswith("LinearRegression")
    assert lines[8].startswith("DecisionTreeRegressor")
    assert lines[9].startswith("RandomForestRegressor")
    assert lines[10].startswith("Done training.")
    assert lines[11].startswith("Model:")
    assert lines[12].startswith("R2 score:")
    assert lines[13].startswith("Model:")
    assert lines[14].startswith("R2 score:")
    assert lines[15].startswith("Model:")
    assert lines[16].startswith("R2 score:")


@pytest.fixture()
def cleanup():
    yield
    os.system("truncate -s 0 log_file.txt")
