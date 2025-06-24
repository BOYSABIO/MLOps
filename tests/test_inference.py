"""Tests for model loading and prediction functions in inference.py."""

import os
import torch
from src.inference.inference import load_trained_model, predict_digits
from src.model.model import CNNModel


def test_load_trained_model():
    """Test if model loads successfully and returns a CNNModel instance."""
    # Save a temporary model
    model = CNNModel()
    temp_path = "temp_model.pth"
    torch.save(model.state_dict(), temp_path)

    # Load the model
    loaded_model = load_trained_model(model_path=temp_path, device="cpu")
    assert isinstance(loaded_model, CNNModel)
    assert not loaded_model.training  # model.eval() should be set

    os.remove(temp_path)


def test_predict_digit_output_type():
    """Test if predict_digit returns an integer prediction."""
    model = CNNModel()
    model.eval()
    dummy_input = torch.rand((1, 1, 28, 28))
    output = predict_digit(model, dummy_input, device="cpu")
    assert isinstance(output, int)


def test_predict_digit_output_range():
    """Test if predict_digit returns a digit in valid range (0–9)."""
    model = CNNModel()
    model.eval()
    dummy_input = torch.rand((1, 1, 28, 28))
    output = predict_digit(model, dummy_input, device="cpu")
    assert 0 <= output <= 9
