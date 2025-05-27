import logging
import numpy as np
from tensorflow.keras.utils import to_categorical

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_images(x):
    logger.info("Normalizing image pixel values to range [0, 1]")
    return np.array(x, dtype="float32") / 255.0

def reshape_images(x):
    logger.info(f"Reshaping images to shape: (-1, 28, 28, 1)")
    return np.reshape(x, (-1, 28, 28, 1))

def one_hot_encode(y):
    logger.info("One-hot encoding labels")
    return to_categorical(y, num_classes=10)

def preprocess_data(x, y, reshape=True):
    logger.info("Starting preprocessing pipeline...")
    x = normalize_images(x)
    if reshape:
        x = reshape_images(x)
    y = one_hot_encode(y)
    logger.info("Preprocessing complete.")
    return x, y
