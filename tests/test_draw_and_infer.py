"""
Tests for preprocess_canvas and save_prediction_image in draw_and_infer.py.
"""

import os
from datetime import datetime
from unittest.mock import patch

import numpy as np
import torch
import cv2

from src.draw_and_infer import preprocess_canvas, save_prediction_image


def test_preprocess_canvas_output_shapes():
    """Check tensor shape and image size returned from preprocess_canvas."""
    canvas = np.zeros((280, 280, 3), dtype=np.uint8)
    cv2.rectangle(
        canvas, (100, 100), (180, 180), (255, 255, 255), -1
    )  # type: ignore

    tensor, img = preprocess_canvas(canvas)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 1, 28, 28)
    assert img.shape == (28, 28)


def test_save_prediction_image_creates_file():
    """Ensure save_prediction_image saves the file correctly."""
    image = np.ones((28, 28), dtype=np.uint8)

    with patch("src.draw_and_infer.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.now.strftime.return_value = "20240101_120000"

        save_prediction_image(image, 3)
        expected_filename = "data/predictions/digit_3_20240101_120000.png"
        assert os.path.exists(expected_filename)

        os.remove(expected_filename)


# Optional cleanup if test crashes
if os.path.exists("data/predictions"):
    for f in os.listdir("data/predictions"):
        if f.startswith("digit_"):
            os.remove(os.path.join("data/predictions", f))
