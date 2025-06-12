import os
import logging
import torch
from torchvision import datasets
import numpy as np

logging.basicConfig(level=logging.INFO)


def validate_mnist_format(x_train, y_train, x_test, y_test):
    """
    Validates that the MNIST data format matches the expected format.
    """
    # Check shapes
    assert x_train.shape == (60000, 28, 28), f"Expected x_train shape (60000, 28, 28), got {x_train.shape}"
    assert y_train.shape == (60000,), f"Expected y_train shape (60000,), got {y_train.shape}"
    assert x_test.shape == (10000, 28, 28), f"Expected x_test shape (10000, 28, 28), got {x_test.shape}"
    assert y_test.shape == (10000,), f"Expected y_test shape (10000,), got {y_test.shape}"

    # Check data types
    assert x_train.dtype == np.uint8, f"Expected x_train dtype uint8, got {x_train.dtype}"
    assert y_train.dtype == np.uint8, f"Expected y_train dtype uint8, got {y_train.dtype}"
    assert x_test.dtype == np.uint8, f"Expected x_test dtype uint8, got {x_test.dtype}"
    assert y_test.dtype == np.uint8, f"Expected y_test dtype uint8, got {y_test.dtype}"

    # Check value ranges
    assert x_train.min() >= 0 and x_train.max() <= 255, "x_train values should be in range [0, 255]"
    assert x_test.min() >= 0 and x_test.max() <= 255, "x_test values should be in range [0, 255]"
    assert y_train.min() >= 0 and y_train.max() <= 9, "y_train values should be in range [0, 9]"
    assert y_test.min() >= 0 and y_test.max() <= 9, "y_test values should be in range [0, 9]"

    logging.info("MNIST data format validation passed")


def load_data():
    """
    Checks if data is already stored locally and loads the tuple
    (x_train, y_train), (x_test, y_test).
    If no data is present locally, it downloads, saves, and loads the data
    from torchvision datasets.
    """
    raw_dir = "data/raw"
    file_names = ["x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy"]

    try:
        if all(
            os.path.isfile(os.path.join(raw_dir, fname))
            for fname in file_names
        ):
            logging.info("Loading MNIST from local .npy files...")
            try:
                x_train = np.load(os.path.join(raw_dir, "x_train.npy"))
                y_train = np.load(os.path.join(raw_dir, "y_train.npy"))
                x_test = np.load(os.path.join(raw_dir, "x_test.npy"))
                y_test = np.load(os.path.join(raw_dir, "y_test.npy"))
            except Exception as e:
                logging.error("Failed to load .npy files", exc_info=True)
                raise RuntimeError("Corrupted or unreadable .npy files") from e
        else:
            logging.info("Downloading MNIST from torchvision...")
            try:
                # Download MNIST dataset
                train_dataset = datasets.MNIST(
                    root='data/raw',
                    train=True,
                    download=True
                )
                test_dataset = datasets.MNIST(
                    root='data/raw',
                    train=False,
                    download=True
                )

                # Convert to numpy arrays
                x_train = train_dataset.data.numpy()
                y_train = train_dataset.targets.numpy()
                x_test = test_dataset.data.numpy()
                y_test = test_dataset.targets.numpy()

                # Validate data format
                validate_mnist_format(x_train, y_train, x_test, y_test)

                # Save the data
                os.makedirs(raw_dir, exist_ok=True)
                np.save(os.path.join(raw_dir, "x_train.npy"), x_train)
                np.save(os.path.join(raw_dir, "y_train.npy"), y_train)
                np.save(os.path.join(raw_dir, "x_test.npy"), x_test)
                np.save(os.path.join(raw_dir, "y_test.npy"), y_test)

            except Exception as e:
                logging.error("Failed to download MNIST data", exc_info=True)
                raise RuntimeError("Could not download MNIST dataset") from e

        return (x_train, y_train), (x_test, y_test)

    except Exception as e:
        logging.error("Error during MNIST data loading process", exc_info=True)
        raise RuntimeError("Failed to load MNIST dataset") from e


def save_raw_data(x_train, y_train, x_test, y_test, save_dir="data/raw"):
    """
    Saves MNIST data in raw format as 4 files for x/y train and test.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "x_train.npy"), x_train)
        np.save(os.path.join(save_dir, "y_train.npy"), y_train)
        np.save(os.path.join(save_dir, "x_test.npy"), x_test)
        np.save(os.path.join(save_dir, "y_test.npy"), y_test)
        logging.info("Saved raw data to %s", save_dir)
    except Exception as e:
        logging.error("Failed to save MNIST data to disk", exc_info=True)
        raise RuntimeError("Failed to save MNIST dataset") from e
