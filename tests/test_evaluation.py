"""Tests for evaluation functions: evaluate_model and plot_confusion_matrix."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from src.evaluation.evaluation import evaluate_model, plot_confusion_matrix
from src.model.model import CNNModel


def test_evaluate_model_accuracy_and_cm():
    """Check if evaluate_model returns valid accuracy and confusion matrix."""
    model = CNNModel()
    model.eval()

    x_dummy = torch.rand((10, 1, 28, 28))
    y_dummy = torch.randint(0, 10, (10,))

    acc, cm = evaluate_model(model, x_dummy, y_dummy)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
    assert isinstance(cm, np.ndarray)
    assert cm.shape[0] == cm.shape[1], "Confusion matrix must be square"


def test_plot_confusion_matrix_creates_file(tmp_path):
    """Verify plot_confusion_matrix creates an output file."""
    cm = np.array([[5, 2], [1, 7]])
    labels = ["Cat", "Dog"]
    output_path = tmp_path / "cm_test.png"

    plot_confusion_matrix(cm, labels=labels, save_path=str(output_path))

    assert os.path.exists(output_path), "Plot was not saved"


def test_plot_confusion_matrix_default_labels(tmp_path):
    """Verify it works with default label names (no label list)."""
    cm = np.eye(3, dtype=int)
    output_path = tmp_path / "cm_default.png"
    plot_confusion_matrix(cm, save_path=str(output_path))
    assert os.path.exists(output_path)


def test_evaluate_model_single_output_warning(caplog):
    """Test warning for single output model."""
    class DummySingleOutputModel(torch.nn.Module):
        """A dummy model that returns single output values.

        This model is used to test the warning behavior when a model outputs
        single values instead of multi-class logits.
        """

        def forward(self, x):
            """Forward pass that returns random single output values.
            Args:
                x: Input tensor

            Returns:
                Random tensor with shape (batch_size, 1)
            """
            return torch.rand((x.shape[0], 1))  # Single output

    model = DummySingleOutputModel()
    x_dummy = torch.rand((10, 1, 28, 28))
    y_dummy = torch.randint(0, 2, (10,))

    with caplog.at_level(logging.WARNING):
        acc, cm = evaluate_model(model, x_dummy, y_dummy)
        assert "Model output may not be multi-class logits" in caplog.text
        assert isinstance(acc, float)
        assert isinstance(cm, np.ndarray)


def test_evaluate_model_exception_handling():
    """Test exception handling in model evaluation."""
    class BrokenModel(torch.nn.Module):
        """A model that intentionally raises an error.

        This model is used to verify that the evaluation function properly
        handles and reports model errors.
        """

        def forward(self, x):
            """Forward pass that always raises a ValueError.

            Args:
                x: Input tensor

            Raises:
                ValueError: Always raised to test error handling
            """
            raise ValueError("Intentional failure")

    model = BrokenModel()
    x_dummy = torch.rand((10, 1, 28, 28))
    y_dummy = torch.randint(0, 10, (10,))

    with pytest.raises(RuntimeError, match="Failed to evaluate model"):
        evaluate_model(model, x_dummy, y_dummy)


def test_plot_confusion_matrix_raises_on_save_error(monkeypatch):
    """Test error handling when saving confusion matrix plot fails."""
    cm = np.array([[1, 0], [0, 1]])

    def mock_savefig(*args, **kwargs):
        raise IOError("Mocked save error")

    monkeypatch.setattr(plt, "savefig", mock_savefig)

    with pytest.raises(
        RuntimeError,
        match="Failed to generate confusion matrix plot"
    ):
        plot_confusion_matrix(cm)
