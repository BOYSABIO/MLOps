import numpy as np
import logging

def validate_data(x: np.ndarray, y: np.ndarray, expected_shape=(28, 28), num_classes=10):
    """
    Validates input image and label data.

    Args:
        x: Feature matrix (images), shape (N, H, W)
        y: Labels (either ints or one-hot), shape (N,) or (N, num_classes)
        expected_shape: Tuple of expected image dimensions (H, W)
        num_classes: Expected number of classes

    Raises:
        ValueError if validation fails
    """
    logging.info("Validating input data...")

    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("Inputs must be numpy arrays")

    if x.ndim != 3:
        raise ValueError(f"x should have 3 dimensions (N, H, W), got {x.ndim}")

    if x.shape[1:] != expected_shape:
        raise ValueError(f"Expected image shape {expected_shape}, got {x.shape[1:]}")

    if y.ndim == 1:
        pass  # y is class index
    elif y.ndim == 2:
        if y.shape[1] != num_classes:
            raise ValueError(f"Expected {num_classes} one-hot classes, got {y.shape[1]}")
        if not np.all(np.logical_or(y == 0, y == 1)):
            raise ValueError("One-hot labels must be 0 or 1 only")
    else:
        raise ValueError("y should be 1D or 2D (for one-hot)")

    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("Found NaNs in data")

    if not np.isfinite(x).all():
        raise ValueError("Found non-finite values in x")

    logging.info("Data validation passed.")