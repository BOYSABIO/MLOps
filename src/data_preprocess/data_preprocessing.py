import logging
import os
import numpy as np
from tensorflow.keras.utils import to_categorical

# Configure logging
logger = logging.getLogger(__name__)


def normalize_images(images):
    """
    Normalizes image pixel values to the range [0, 1].

    Converts the input array to float32 and scales pixel values 
    by dividing by 255.
    """
    logger.info("Normalizing image pixel values to range [0, 1]")
    return np.array(images, dtype="float32") / 255.0


def reshape_images(images):
    """
    Reshapes the input array to shape (-1, 28, 28, 1).

    This is typically used to format grayscale images for CNN input.
    """
    logger.info("Reshaping images to shape: (-1, 28, 28, 1)")
    return np.reshape(images, (-1, 28, 28, 1))


def one_hot_encode(label):
    """
    Converts class labels into one-hot encoded vectors.

    Assumes labels are integers in range [0, 9] for 10 classes.
    """
    logger.info("One-hot encoding labels")
    return to_categorical(label, num_classes=10)


def preprocess_data(images, labels, reshape=True):
    """
    Applies a preprocessing pipeline to the input image and label data.

    - Normalizes image pixel values to the range [0, 1]
    - Optionally reshapes images to (28, 28, 1) for CNN input
    - One-hot encodes the labels

    Returns:
        Tuple (processed_images, encoded_labels)
    """
    logger.info("Starting preprocessing pipeline...")
    processed_images = normalize_images(images)
    if reshape:
        processed_images = reshape_images(processed_images)
    encoded_labels = one_hot_encode(labels)
    logger.info("Preprocessing complete.")
    return processed_images, encoded_labels    


def save_preprocessed_data(images, labels, output_dir):
    """
    Saves the preprocessed image and label arrays as .npy files.

    Args:
        images: Preprocessed image array.
        labels: One-hot encoded label array.
        output_dir: Directory path where files will be saved.
    """
    logger.info(f"Saving preprocessed data to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    images_path = os.path.join(output_dir, "images.npy")
    labels_path = os.path.join(output_dir, "labels.npy")

    np.save(images_path, images)
    np.save(labels_path, labels)
    
    logger.info(f"Images saved to: {images_path}")
    logger.info(f"Labels saved to: {labels_path}")
