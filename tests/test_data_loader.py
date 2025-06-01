import pytest
import numpy as np
from src.data_load.data_loader import load_data
import os


def test_load_data_shapes():
    (x_train, y_train), (x_test, y_test) = load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

def test_saved_files_exist():
    expected_files = [
        "data/raw/x_train.npy",
        "data/raw/y_train.npy",
        "data/raw/x_test.npy",
        "data/raw/y_test.npy"
    ]
    for file in expected_files:
        assert os.path.exists(file), f"{file} was not found"

