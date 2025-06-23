"""MLflow entry point for feature extraction and visualization."""

import logging
import os
import sys

import click
import numpy as np
import torch

from src.model.model import load_model
from src.features.features import extract_embeddings, tsne_plot, pca_plot
from src.utils.logging_config import get_logger

# Add the src directory to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


@click.command()
@click.option("--model-path", required=True, help="Path to the trained model")
@click.option("--test-images-path", required=True, help="Path to test images")
@click.option("--test-labels-path", required=True, help="Path to test labels")
@click.option("--output-dir", required=True,
              help="Directory to save plots and embeddings")
def main(model_path, test_images_path, test_labels_path, output_dir):
    """
    MLflow entry point to extract CNN embeddings and visualize them.
    """
    try:
        logger.info("Step: Feature Extraction executed.")
        logger.info("Model path: %s", model_path)
        logger.info("Saving outputs to: %s", output_dir)

        model = load_model(model_path)
        x_test = np.load(test_images_path)
        y_test = np.load(test_labels_path)

        x_tensor = torch.tensor(
            x_test[:1000], dtype=torch.float32).permute(0, 3, 1, 2)
        y_tensor = torch.tensor(
            np.argmax(y_test[:1000], axis=1), dtype=torch.long)

        embeddings = extract_embeddings(model, x_tensor)

        os.makedirs(output_dir, exist_ok=True)
        np.savez(
            os.path.join(output_dir, "embeddings.npz"),
            embeddings=embeddings,
            labels=y_tensor.numpy()
        )

        tsne_plot(embeddings, y_tensor.numpy(),
                  os.path.join(output_dir, "tsne.png"))
        pca_plot(embeddings, y_tensor.numpy(),
                 os.path.join(output_dir, "pca.png"))

        logger.info("✅ Feature extraction and visualization completed.")
    except Exception as e:
        logger.error("❌ Feature extraction failed", exc_info=True)
        raise RuntimeError("features failed") from e


if __name__ == "__main__":
    main()
