"""Tests for evaluation functions: evaluate_model and plot_confusion_matrix."""

import os
import torch
import numpy as np
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
