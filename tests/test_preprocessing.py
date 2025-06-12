"""Tests for preprocessing functions in data_preprocessing.py."""

import os
import tempfile
import numpy as np

from src.data_preprocess.data_preprocessing import (
    normalize_images,
    reshape_images,
    one_hot_encode,
    preprocess_data,
    save_preprocessed_data
)


def test_normalize_images():
    """Check normalization scales pixel values to [0, 1] with float32 dtype."""
    dummy = np.array([[0, 255]], dtype="uint8")
    result = normalize_images(dummy)
    assert (
        np.all(result >= 0.0) and np.all(result <= 1.0)
    ), "Values not in [0, 1]"
    assert result.dtype == np.float32


def test_reshape_images():
    """Verify that reshape adds a 4th channel dimension."""
    dummy = np.random.rand(10, 28, 28)
    reshaped = reshape_images(dummy)
    assert reshaped.shape == (10, 28, 28, 1), f"Got shape {reshaped.shape}"
    assert reshaped.ndim == 4


def test_one_hot_encode():
    """Ensure one-hot encoding works correctly for digit labels."""
    labels = np.array([0, 1, 9])
    encoded = one_hot_encode(labels)
    assert encoded.shape == (3, 10), f"Got shape {encoded.shape}"
    assert np.all(encoded.sum(axis=1) == 1), "Each row should sum to 1"
    assert np.array_equal(np.argmax(encoded, axis=1), labels)


def test_preprocess_data_with_reshape():
    """Test preprocessing pipeline with reshaping enabled."""
    x = np.random.randint(0, 256, size=(5, 28, 28), dtype="uint8")
    y = np.array([1, 3, 5, 7, 9])
    x_out, y_out = preprocess_data(x, y, reshape=True)
    assert x_out.shape == (5, 28, 28, 1)
    assert x_out.dtype == np.float32
    assert y_out.shape == (5, 10)


def test_preprocess_data_without_reshape():
    """Test preprocessing pipeline without reshaping."""
    x = np.random.randint(0, 256, size=(5, 28, 28), dtype="uint8")
    y = np.array([0, 2, 4, 6, 8])
    x_out, y_out = preprocess_data(x, y, reshape=False)
    assert x_out.shape == (5, 28, 28)
    assert y_out.shape == (5, 10)


def test_save_preprocessed_data_creates_files():
    """Ensure save_preprocessed_data writes .npy files to the given directory.

    Verifies that the function creates and saves both image and label files.
    """
    x = np.random.rand(3, 28, 28, 1)
    y = np.eye(10)[np.array([0, 1, 2])]

    with tempfile.TemporaryDirectory() as tmpdir:
        save_preprocessed_data(x, y, tmpdir, "test")
        x_path = os.path.join(tmpdir, "test_images.npy")
        y_path = os.path.join(tmpdir, "test_labels.npy")

        assert os.path.isfile(x_path), "Images file not created"
        assert os.path.isfile(y_path), "Labels file not created"

        x_loaded = np.load(x_path)
        y_loaded = np.load(y_path)
        np.testing.assert_array_equal(x, x_loaded)
        np.testing.assert_array_equal(y, y_loaded)
