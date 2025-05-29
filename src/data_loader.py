import logging
from tensorflow.keras.datasets import mnist # type: ignore
import numpy as np
import os

def load_data():
    """
    Checks if data is already stored locally and loads the tuple (x_train, y_train), (x_test, y_test)

    If no data is present locally, it downloads, saves, and loads the data from keras datasets.
    """
    raw_dir = "data/raw"
    try:
        if all(os.path.exists(os.path.join(raw_dir, fname)) for fname in ["x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy"]):
            logging.info("Loading MNIST from local .npy files...")
            x_train = np.load(os.path.join(raw_dir, "x_train.npy"))
            y_train = np.load(os.path.join(raw_dir, "y_train.npy"))
            x_test = np.load(os.path.join(raw_dir, "x_test.npy"))
            y_test = np.load(os.path.join(raw_dir, "y_test.npy"))
        else:
            logging.info("Downloading MNIST from Keras...")
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            save_raw_data(x_train, y_train, x_test, y_test)
        return (x_train, y_train), (x_test, y_test)
    except Exception as e:
        logging.error("Error loading MNIST data", exc_info=True)
        raise RuntimeError("Failed to load MNIST dataset") from e
    
def save_raw_data(x_train, y_train, x_test, y_test, save_dir = "data/raw"):
    """
    Saves MNIST data in raw format as 4 files for xy train and test.
    """
    os.makedirs(save_dir, exist_ok = True)
    np.save(os.path.join(save_dir, "x_train.npy"), x_train)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "x_test.npy"), x_test)
    np.save(os.path.join(save_dir, "y_test"), y_test)
    logging.info(f"Saved raw data to {save_dir}")