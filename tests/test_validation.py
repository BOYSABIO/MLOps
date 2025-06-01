import pytest
import numpy as np
from src.data_validation.validation import validate_data


def test_valid_data_passes():
    x = np.random.rand(100, 28, 28)
    y = np.eye(10)[np.random.randint(0, 10, 100)]  # one-hot
    validate_data(x, y)

def test_wrong_shape_raises():
    x = np.random.rand(100, 32, 32)
    y = np.eye(10)[np.random.randint(0, 10, 100)]
    with pytest.raises(ValueError):
        validate_data(x, y)

def test_wrong_num_classes_raises():
    x = np.random.rand(100, 28, 28)
    y = np.eye(8)[np.random.randint(0, 8, 100)]
    with pytest.raises(ValueError):
        validate_data(x, y)

def test_nan_raises():
    x = np.random.rand(100, 28, 28)
    x[0, 0, 0] = np.nan
    y = np.eye(10)[np.random.randint(0, 10, 100)]
    with pytest.raises(ValueError):
        validate_data(x, y)