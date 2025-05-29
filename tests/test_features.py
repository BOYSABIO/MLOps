import numpy as np
from src.features import flatten_images

def test_flatten_images():
    # Create dummy input of shape (batch_size, height, width)
    dummy_images = np.random.rand(5, 28, 28)

    # Flatten the images
    flattened = flatten_images(dummy_images)

    # Assertions
    assert flattened.shape == (5, 784), f"Expected shape (5, 784), got {flattened.shape}"
    assert flattened.ndim == 2, "Flattened output should be 2D"
    assert np.allclose(flattened[0], dummy_images[0].flatten()), "Flattened content does not match original"
