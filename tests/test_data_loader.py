import pytest
from src.data_loader import load_data

def test_load_data_shape():
    (x_train, y_train), (x_test, y_test) = load_data()
    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_train.shape[0]
    assert x_train.shape[1:] == (28, 28)
    assert y_test.shape[1:] == (28, 28)