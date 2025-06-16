"""Tests for the MNIST data loader including local and fallback behaviors."""

# pylint: disable=redefined-outer-name
import os
import shutil
from unittest import mock

import numpy as np
import pytest

from src.data_load.data_loader import load_data, save_raw_data


@pytest.fixture(scope="function")
def cleanup_raw_dir():
    """Ensure the data/raw directory is clean before and after each test."""
    raw_dir = "data/raw"
    if os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
    os.makedirs(raw_dir, exist_ok=True)
    yield
    shutil.rmtree(raw_dir)


def test_load_data_shapes():
    """Basic shape check on returned MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)


def test_saved_files_exist():
    """Ensure MNIST .npy files exist after load."""
    load_data()
    expected_files = [
        "data/raw/x_train.npy",
        "data/raw/y_train.npy",
        "data/raw/x_test.npy",
        "data/raw/y_test.npy",
    ]
    for file in expected_files:
        assert os.path.exists(file), f"{file} was not found"


# pylint: disable=unused-argument
def test_save_raw_data_creates_files(cleanup_raw_dir):
    """Check that save_raw_data properly saves all 4 files."""
    dummy = np.zeros((10, 28, 28), dtype=np.uint8)
    labels = np.zeros((10,), dtype=np.uint8)
    save_raw_data(dummy, labels, dummy, labels)

    for fname in ["x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy"]:
        path = os.path.join("data/raw", fname)
        assert os.path.isfile(path)


# pylint: disable=unused-argument
def test_load_data_download_called_if_missing(monkeypatch, cleanup_raw_dir):
    """Simulate empty directory and mock keras mnist.load_data call."""
    dummy_x = np.zeros((60000, 28, 28), dtype=np.uint8)
    dummy_y = np.zeros((60000,), dtype=np.uint8)

    mock_mnist = mock.Mock(
        return_value=((dummy_x, dummy_y), (dummy_x[:10000], dummy_y[:10000]))
    )
    monkeypatch.setattr(
        "src.data_load.data_loader.mnist.load_data",
        mock_mnist
    )

    data = load_data()
    assert mock_mnist.called
    assert data[0][0].shape == (60000, 28, 28)


# pylint: disable=unused-argument
def test_load_data_raises_on_corrupt_files(cleanup_raw_dir):
    """Test error handling when .npy files are corrupted or unreadable."""
    with open("data/raw/x_train.npy", "wb") as f:
        f.write(b"not a real numpy file")
    open("data/raw/y_train.npy", "wb").close()
    open("data/raw/x_test.npy", "wb").close()
    open("data/raw/y_test.npy", "wb").close()

    with pytest.raises(RuntimeError) as exc_info:
        load_data()

    print(f"ACTUAL MESSAGE: {str(exc_info.value)}")
    assert "Failed to load MNIST dataset" in str(exc_info.value)
    assert "Corrupted or unreadable" in str(exc_info.value.__cause__)


def test_save_raw_data_raises_on_failure(monkeypatch):
    """Simulate failure to save numpy arrays and check error handling."""
    def mock_save(*args, **kwargs):
        raise IOError("Disk write failed")
    monkeypatch.setattr("numpy.save", mock_save)

    dummy = np.zeros((10, 28, 28), dtype=np.uint8)
    labels = np.zeros((10,), dtype=np.uint8)

    with pytest.raises(RuntimeError) as exc_info:
        save_raw_data(dummy, labels, dummy, labels)

    assert "Failed to save MNIST dataset" in str(exc_info.value)


def test_load_data_raises_on_failed_local_load(monkeypatch):
    """Simulate failure when loading existing .npy files."""
    monkeypatch.setattr("os.path.isfile", lambda path: True)

    def mock_load(path):
        raise IOError("Load failed")
    monkeypatch.setattr("numpy.load", mock_load)

    with pytest.raises(RuntimeError) as exc_info:
        load_data()

    assert "Failed to load MNIST dataset" in str(exc_info.value)
    assert "Corrupted or unreadable" in str(exc_info.value.__cause__)
