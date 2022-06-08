"""
This module contains unit tests for housing_packaged/train.py.
"""
import os

import housing_packaged.train as train
from housing_packaged.logger import configure_logger
from sklearn.svm import SVR

args = train.parse_args()
logger = configure_logger()


def test_parse_args():
    """
    Tests parse_args function.
    """
    assert args.dataset == "data/processed/housing_train.csv"
    assert args.models == "data/models/"
    assert args.log_level == "DEBUG"
    assert not args.no_console_log
    assert args.log_path == "log_file.txt"


def test_load_data():
    """
    Tests load_data function.
    """
    X, y = train.load_data(args.dataset)
    assert len(X) == len(y)
    assert "median_house_value" not in X.columns
    assert not X.isna().sum().sum()
    assert len(y.shape) == 1


def test_save():
    """
    Tests save_model function.
    """
    svr = SVR()
    train.save_model(svr, args.models)
    name = type(svr).__name__
    assert os.path.isfile(f"{args.models}/{name}.pkl")
    os.remove(f"{args.models}/{name}.pkl")
