import argparse
import click
import logging
import numpy as np
import os
import sys
import torch

# Make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.model import load_model
from src.features.features import extract_embeddings, tsne_plot, pca_plot
from src.utils.logging_config import get_logger

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

@click.command()
@click.option("--model-path", required=True, help="Path to the trained model")
@click.option("--test-images-path", required=True, help="Path to test images")
@click.option("--test-labels-path", required=True, help="Path to test labels")
@click.option("--output-dir", required=True, help="Directory to save feature visualizations")
def main(model_path, test_images_path, test_labels_path, output_dir):
    """
    MLflow entry point para extraer y visualizar características
    """
    try:
        logger.info("Step: Feature Extraction executed.")
        logger.info(f"Model: {model_path}")
        logger.info(f"Test images: {test_images_path}")
        logger.info(f"Output directory: {output_dir}")

        # Load model and test data
        model = load_model(model_path)
        x_test = np.load(test_images_path)
        y_test = np.load(test_labels_path)

        # Preprocess for torch input
        x_tensor = torch.tensor(x_test[:1000], dtype=torch.float32).permute(0, 3, 1, 2)
        y_tensor = torch.tensor(np.argmax(y_test[:1000], axis=1), dtype=torch.long)

        # Extract embeddings
        embeddings = extract_embeddings(model, x_tensor)

        # Save embeddings
        os.makedirs(output_dir, exist_ok=True)
        np.savez(
            os.path.join(output_dir, "embeddings.npz"),
            embeddings=embeddings,
            labels=y_tensor.numpy()
        )

        # Plot
        tsne_plot(embeddings, y_tensor.numpy(), save_path=os.path.join(output_dir, "tsne_plot.png"))
        pca_plot(embeddings, y_tensor.numpy(), save_path=os.path.join(output_dir, "pca_plot.png"))

        logger.info("✅ Feature extraction completed successfully.")
    except Exception as e:
        logger.error("❌ Feature Extraction failed", exc_info=True)
        raise RuntimeError("features failed") from e

if __name__ == "__main__":
    main()
