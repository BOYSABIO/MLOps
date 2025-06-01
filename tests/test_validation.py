"""Tests for the validate_data function in data_validation module."""

import numpy as np
import pytest
from src.data_validation.validation import validate_data


def test_valid_data_passes():
    """Ensure validation passes for correctly shaped and clean data."""
    x = np.random.rand(100, 28, 28)
    y = np.eye(10)[np.random.randint(0, 10, 100)]  # one-hot
    validate_data(x, y)


def test_wrong_shape_raises():
    """Check that wrong image shape raises a ValueError."""
    x = np.random.rand(100, 32, 32)
    y = np.eye(10)[np.random.randint(0, 10, 100)]
    with pytest.raises(ValueError):
        validate_data(x, y)


def test_wrong_num_classes_raises():
    """Check that using less than 10 output classes raises a ValueError."""
    x = np.random.rand(100, 28, 28)
    y = np.eye(8)[np.random.randint(0, 8, 100)]
    with pytest.raises(ValueError):
        validate_data(x, y)


def test_nan_raises():
    """Ensure data with NaNs raises a ValueError."""
    x = np.random.rand(100, 28, 28)
    x[0, 0, 0] = np.nan
    y = np.eye(10)[np.random.randint(0, 10, 100)]
    with pytest.raises(ValueError):
        validate_data(x, y)
