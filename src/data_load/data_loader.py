import os
import logging
from tensorflow.keras.datasets import mnist
import numpy as np

logging.basicConfig(level=logging.INFO)


def load_data(raw_data_path="data/raw"):
    """
    Checks if data is already stored locally and loads the tuple
    (x_train, y_train), (x_test, y_test).
    If no data is present locally, it downloads, saves, and loads the data
    from keras datasets.
    """
    raw_dir = raw_data_path
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
            logging.info("Downloading MNIST from Keras...")
            try:
                (x_train, y_train), (x_test, y_test) = mnist.load_data()
                save_raw_data(x_train, y_train, x_test, y_test, raw_dir)
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
