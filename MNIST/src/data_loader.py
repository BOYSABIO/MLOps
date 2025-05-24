import logging
from tensorflow.keras.datasets import mnist

def load_data():
    """
    Loads the MNIST dataset from keras and returns train and test sets.

    Returns:
        Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test)
    """
    try:
        logging.info("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        logging.info(f"Data loaded successfully. Train shape: {x_train.shape}, Test shape {x_test.shape}")
        return (x_train, y_train), (x_test, y_test)
    except Exception as e:
        logging.error("Error loading MNIST data", exc_info = True)
        raise RuntimeError("Failed to load MNIST dataset") from e