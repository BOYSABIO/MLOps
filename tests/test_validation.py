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


def test_non_integer_1d_labels_raise():
    """1D labels must be integers."""
    x = np.random.rand(100, 28, 28)
    y = np.array([str(i) for i in range(100)])  # strings instead of ints
    with pytest.raises(
        ValueError,
        match="1D label array should contain integers"
    ):
        validate_data(x, y)


def test_invalid_one_hot_values():
    """One-hot labels must contain only 0 or 1."""
    x = np.random.rand(100, 28, 28)
    y = np.eye(10)[np.random.randint(0, 10, 100)]
    y[0, 0] = 0.5  # break one-hot constraint
    with pytest.raises(
        ValueError,
        match="One-hot labels must contain only 0 or 1"
    ):
        validate_data(x, y)


def test_one_hot_row_sum_not_one():
    """Each one-hot row must sum to 1."""
    x = np.random.rand(100, 28, 28)
    y = np.eye(10)[np.random.randint(0, 10, 100)]
    y[0] = 0  # zero row, sum is 0
    with pytest.raises(
        ValueError,
        match="Each one-hot encoded label must sum to 1"
    ):
        validate_data(x, y)


def test_non_numpy_input_raises():
    """x and y must be numpy arrays."""
    x = [[0.0] * 28] * 100
    y = [1] * 100
    with pytest.raises(ValueError, match="Both x and y must be numpy arrays"):
        validate_data(x, y)


def test_inf_values_raise():
    """Non-finite values like inf should raise error."""
    x = np.random.rand(100, 28, 28)
    x[0, 0, 0] = np.inf
    y = np.eye(10)[np.random.randint(0, 10, 100)]
    with pytest.raises(ValueError, match="non-finite values"):
        validate_data(x, y)
