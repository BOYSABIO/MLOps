import logging
import os
import numpy as np
from tensorflow.keras.utils import to_categorical  # type: ignore

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def normalize_images(images):
    """
    Normalizes image pixel values to the range [0, 1].
    """
    try:
        logger.info("Normalizing image pixel values to range [0, 1]")
        return np.array(images, dtype="float32") / 255.0
    except Exception as e:
        logger.error("Failed to normalize images", exc_info=True)
        raise RuntimeError("Normalization failed") from e


def reshape_images(images):
    """
    Reshapes the input array to shape (-1, 28, 28, 1).
    """
    try:
        logger.info("Reshaping images to shape: (-1, 28, 28, 1)")
        return np.reshape(images, (-1, 28, 28, 1))
    except Exception as e:
        logger.error("Failed to reshape images", exc_info=True)
        raise ValueError("Invalid image shape for reshaping") from e


def one_hot_encode(label):
    """
    Converts class labels into one-hot encoded vectors.
    """
    try:
        logger.info("One-hot encoding labels")
        return to_categorical(label, num_classes=10)
    except Exception as e:
        logger.error("Failed to one-hot encode labels", exc_info=True)
        raise RuntimeError("One-hot encoding failed") from e


def preprocess_data(images, labels, reshape=True):
    """
    Applies a preprocessing pipeline to the input image and label data.
    """
    try:
        logger.info("Starting preprocessing pipeline...")
        processed_images = normalize_images(images)
        if reshape:
            processed_images = reshape_images(processed_images)
        encoded_labels = one_hot_encode(labels)
        logger.info("Preprocessing complete.")
        return processed_images, encoded_labels
    except Exception as e:
        logger.error("Preprocessing pipeline failed", exc_info=True)
        raise RuntimeError("Data preprocessing failed") from e


def save_preprocessed_data(images, labels, output_dir, data_type):
    """
    Saves the preprocessed image and label arrays as .npy files.
    """
    try:
        logger.info(
            "Saving preprocessed %s data to: %s", data_type, output_dir
        )
        os.makedirs(output_dir, exist_ok=True)

        images_path = os.path.join(output_dir, f"{data_type}_images.npy")
        labels_path = os.path.join(output_dir, f"{data_type}_labels.npy")

        np.save(images_path, images)
        np.save(labels_path, labels)

        logger.info("Images saved to: %s", images_path)
        logger.info("Labels saved to: %s", labels_path)
    except Exception as e:
        logger.error("Failed to save %s data", data_type, exc_info=True)
        raise IOError(f"Could not save {data_type} data") from e
