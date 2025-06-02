"""Test suite for the CNN model implementation.

This module contains unit tests for the CNN model, including:
- Forward pass functionality
- Training loop
- Device management
- Model saving and loading
"""
import os
import tempfile
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model.model import (
    CNNModel,
    train_model,
    get_device,
    save_model,
    load_model,
)


def test_model_forward_pass():
    """Test that the model's forward pass produces correct output shape."""
    model = CNNModel()
    dummy_input = torch.randn(8, 1, 28, 28)
    output = model(dummy_input)
    assert output.shape == (8, 10)


def test_model_training_loop():
    """Test that the model training loop completes successfully."""
    x = torch.rand(16, 1, 28, 28)
    y = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    config = {"model": {"epochs": 1, "learning_rate": 1e-3}}
    model = CNNModel()
    trained = train_model(model, loader, config)

    assert isinstance(trained, CNNModel)


def test_get_device_returns_valid_device():
    """Test that get_device returns a valid torch device."""
    device = get_device()
    assert isinstance(device, torch.device)


def test_save_and_load_model():
    """Test that model can be saved and loaded correctly."""
    model = CNNModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "model.pth")
        save_model(model, save_path)

        assert os.path.exists(save_path)

        loaded_model = load_model(save_path)
        assert isinstance(loaded_model, CNNModel)
