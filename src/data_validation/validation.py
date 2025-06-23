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

import logging
import numpy as np


def validate_data(
    x: np.ndarray,
    y: np.ndarray,
    expected_shape=(28, 28),
    num_classes=10
):    
    try:
        logging.info("Validating input data...")

        # Type checks
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("Both x and y must be numpy arrays")

        # Image shape checks
        if x.ndim != 3:
            raise ValueError(
                f"x should have 3 dimensions (N, H, W), got {x.ndim}"
            )

        if x.shape[1:] != expected_shape:
            raise ValueError(
                f"Expected image shape {expected_shape}, got {x.shape[1:]}"
            )

        # Label checks
        if y.ndim == 1:
            if not np.issubdtype(y.dtype, np.integer):
                raise ValueError("1D label array should contain integers")
        elif y.ndim == 2:
            if y.shape[1] != num_classes:
                raise ValueError(
                    f"Expected one-hot with {num_classes} classes, "
                    f"got {y.shape[1]}"
                )
            if not np.all((y == 0) | (y == 1)):
                raise ValueError("One-hot labels must contain only 0 or 1")
            if not np.allclose(np.sum(y, axis=1), 1):
                raise ValueError("Each one-hot encoded label must sum to 1")
        else:
            raise ValueError(
                "y should be either 1D (class indices) or 2D (one-hot)"
            )

        # Check for NaNs or infinite values
        if np.isnan(x).any() or np.isnan(y).any():
            raise ValueError("Input data contains NaN values")

        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(
                "Input data contains non-finite values (inf or -inf)"
            )

        logging.info("Data validation passed.")

    except Exception as e:
        logging.error("Data validation failed: %s", e, exc_info=True)
        raise
