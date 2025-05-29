import pytest
from src.data_load.data_loader import load_data

def test_load_data_shape():
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Number of samples
    assert x_train.shape[0] == y_train.shape[0] == 60000
    assert x_test.shape[0] == y_test.shape[0] == 10000
    
    # Image dimensions
    assert x_train.shape[1:] == (28, 28)
    assert x_test.shape[1:] == (28, 28)
    
    # Labels should be 1D arrays (if not yet one-hot encoded)
    assert len(y_train.shape) == 1 or y_train.shape[1] == 10  # allow for one-hot too
    assert len(y_test.shape) == 1 or y_test.shape[1] == 10
