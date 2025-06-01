import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import os
import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def get_device():
    """
    Automatically select the appropriate device (MPS for Mac, CUDA for Nvidia, CPU fallback).
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
    """
    logger.info("Extracting features from model...")
    model.eval()
    model.to(device)

    with torch.no_grad():
        x = data.to(device)

        # Same operations as model.forward, but stop before the final output
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
        logger.info(f"Extracted embeddings shape: {embeddings.shape}")
        return embeddings


def tsne_plot(embeddings, labels, save_path="reports/figures/tsne_plot.png"):
    logger.info("Generating t-SNE plot...")
    tsne = TSNE(n_components=2, random_state=42)
    projected = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(projected[:, 0], projected[:, 1], c=labels, cmap="tab10", s=10)
    plt.colorbar(scatter)
    plt.title("t-SNE of CNN Feature Embeddings")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"t-SNE plot saved to {save_path}")


def pca_plot(embeddings, labels, save_path="reports/figures/pca_plot.png"):
    logger.info("Generating PCA plot...")
    pca = PCA(n_components=2)
    projected = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(projected[:, 0], projected[:, 1], c=labels, cmap="tab10", s=10)
    plt.colorbar(scatter)
    plt.title("PCA of CNN Feature Embeddings")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"PCA plot saved to {save_path}")
