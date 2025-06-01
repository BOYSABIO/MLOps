import os

import logging
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_device():
    """
    Automatically select the appropriate device (MPS for Mac, CUDA for Nvidia,
    CPU fallback).
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def extract_embeddings(model, data, device='cpu'):
    """
    Extract intermediate embeddings from CNNModel before classification layer.

    Args:
        model: Trained CNN model
        data: Input tensor
        device: torch.device or str

    Returns:
        np.ndarray: Feature embeddings
    """
    logger.info("Extracting features from model...")
    model.to(device)
    model.eval()

    try:
        with torch.no_grad():
            x = data.to(device)

            # Manually replicate forward pass up to feature vector
            x = F.relu(model.bn1(model.conv1(x)))
            x = model.dropout(x)
            x = F.relu(model.bn2(model.conv2(x)))
            x = model.dropout(x)
            x = F.relu(model.bn3(model.conv3(x)))
            x = model.dropout(x)
            x = model.flatten(x)
            x = F.relu(model.fc1(x))
            x = model.dropout(x)

            embeddings = x.cpu().numpy()

            logger.info("Extracted embeddings shape: %s", embeddings.shape)
            return embeddings
    except Exception as e:
        logger.error("Failed to extract embeddings", exc_info=True)
        raise RuntimeError("Embedding extraction failed") from e


def tsne_plot(embeddings, labels, save_path="reports/figures/tsne_plot.png"):
    """
    Generate and save a t-SNE plot of feature embeddings.

    Args:
        embeddings: 2D numpy array
        labels: 1D list or array of integer labels
        save_path: Where to save the plot
    """
    try:
        logger.info("Generating t-SNE plot...")

        if len(embeddings) != len(labels):
            raise ValueError("Embeddings and labels must have the same length")

        tsne = TSNE(n_components=2, random_state=42)
        projected = tsne.fit_transform(embeddings)

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

        logger.info("t-SNE plot saved to %s", save_path)
    except Exception as e:
        logger.error("Failed to generate t-SNE plot", exc_info=True)
        raise RuntimeError("t-SNE plotting failed") from e


def pca_plot(embeddings, labels, save_path="reports/figures/pca_plot.png"):
    """
    Generate and save a PCA plot of feature embeddings.

    Args:
        embeddings: 2D numpy array
        labels: 1D list or array of integer labels
        save_path: Where to save the plot
    """
    try:
        logger.info("Generating PCA plot...")

        if len(embeddings) != len(labels):
            raise ValueError("Embeddings and labels must have the same length")

        pca = PCA(n_components=2)
        projected = pca.fit_transform(embeddings)

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
        plt.title("PCA of CNN Feature Embeddings")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info("PCA plot saved to %s", save_path)
    except Exception as e:
        logger.error("Failed to generate PCA plot", exc_info=True)
        raise RuntimeError("PCA plotting failed") from e
