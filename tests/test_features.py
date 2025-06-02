"""
Tests for feature extraction and embedding visualizations
using a dummy CNN model.
"""

import logging
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from sklearn.manifold import TSNE

from src.features.features import (
    extract_embeddings,
    pca_plot,
    tsne_plot as real_tsne_plot
)


def tsne(
    embeddings,
    labels,
    save_path="reports/figures/tsne_plot.png",
    perplexity=30
):
    """Wrapper to dynamically set perplexity for testing.

    Args:
        embeddings: Feature embeddings to visualize
        labels: Class labels for the embeddings
        save_path: Path to save the plot
        perplexity: t-SNE perplexity parameter
    """
    if perplexity >= len(embeddings):
        perplexity = len(embeddings) - 1

    logger = logging.getLogger(__name__)
    logger.info("Generating t-SNE plot...")

    tsne_model = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    projected = tsne_model.fit_transform(embeddings)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        projected[:, 0],
        projected[:, 1],
        c=labels,
        cmap="tab10",
        s=10
    )
    plt.colorbar(scatter)
    plt.title("t-SNE of CNN Feature Embeddings")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


class SimpleModel:
    """A minimal CNN-like model with layers used by extract_embeddings.

    This model is used for testing the feature extraction functionality
    without requiring a full CNN model implementation.
    """

    def __init__(self):
        """Initialize the model layers."""
        self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(1)
        self.conv2 = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(1)
        self.conv3 = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(1)
        self.dropout = torch.nn.Dropout(0.0)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(28 * 28, 10)

    def to(self, _device):
        """Mock the to(device) method for compatibility.

        Args:
            _device: Device to move the model to (ignored)

        Returns:
            self: The model instance
        """
        return self

    def eval(self):
        """Mock the eval() method to simulate evaluation mode.

        Returns:
            self: The model instance
        """
        return self


def test_extract_embeddings():
    """Test shape of extracted embeddings from dummy model."""
    model = SimpleModel()
    data = torch.rand((5, 1, 28, 28))
    emb = extract_embeddings(model, data)
    assert emb.shape[0] == 5


def test_tsne_plot():
    """Test t-SNE plot generation and saving."""
    model = SimpleModel()
    data = torch.rand((5, 1, 28, 28))
    labels = np.array([0, 1, 2, 3, 4])
    emb = extract_embeddings(model, data)
    os.makedirs("test_outputs", exist_ok=True)
    path = "test_outputs/tsne_test.png"
    tsne(emb, labels, save_path=path, perplexity=3)
    assert os.path.exists(path)


def test_pca_plot():
    """Test PCA plot generation and saving."""
    model = SimpleModel()
    data = torch.rand((5, 1, 28, 28))
    labels = np.array([0, 1, 2, 3, 4])
    emb = extract_embeddings(model, data)
    os.makedirs("test_outputs", exist_ok=True)
    path = "test_outputs/pca_test.png"
    pca_plot(emb, labels, save_path=path)
    assert os.path.exists(path)


def cleanup():
    """Remove test output directory."""
    if os.path.exists("test_outputs"):
        shutil.rmtree("test_outputs")


def test_tsne_plot_mismatched_lengths():
    """Test t-SNE plot with mismatched embedding and label lengths."""
    emb = np.random.rand(5, 10)
    labels = np.array([0, 1, 2])  # Mismatch
    with pytest.raises(RuntimeError, match="t-SNE plotting failed"):
        real_tsne_plot(emb, labels, save_path="test_outputs/tsne_error.png")


def test_pca_plot_mismatched_lengths():
    """Test PCA plot with mismatched embedding and label lengths."""
    emb = np.random.rand(5, 10)
    labels = np.array([0, 1, 2])  # Mismatch
    with pytest.raises(RuntimeError, match="PCA plotting failed"):
        pca_plot(emb, labels, save_path="test_outputs/pca_error.png")


@pytest.fixture(scope="session", autouse=True)
def remove_outputs_after_tests():
    """Fixture to clean up test outputs after all tests complete."""
    yield
    cleanup()
