"""MLflow entry point for evaluating a trained CNN model."""

import os
import sys
import logging

import click
import numpy as np
import torch

from src.evaluation.evaluation import evaluate_model, plot_confusion_matrix
from src.utils.logging_config import get_logger
from src.model.model import CNNModel

# Add the src directory to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

# Suppress TensorFlow warnings (if applicable)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@click.command()
@click.option("--model-path", required=True, help="Path to the trained model")
@click.option("--test-images-path", required=True, help="Path to test images")
@click.option("--test-labels-path", required=True, help="Path to test labels")
def main(model_path, test_images_path, test_labels_path):
    """
    MLflow entry point para evaluar el modelo.
    """
    try:
        logger.info("Step: Model Evaluation executed.")
        logger.info("Evaluating model: %s", model_path)
        logger.info("Test images: %s", test_images_path)
        logger.info("Test labels: %s", test_labels_path)

        # Load test data
        x_test = np.load(test_images_path)
        y_test = np.load(test_labels_path)

        # Convert to tensors
        x_test_tensor = torch.tensor(
            x_test, dtype=torch.float32).permute(0, 3, 1, 2)
        y_test_tensor = torch.tensor(
            np.argmax(y_test, axis=1), dtype=torch.long)

        # Load model
        model = CNNModel()
        model.load_state_dict(torch.load(model_path))

        # Evaluate model
        accuracy, confusion_matrix = evaluate_model(
            model, x_test_tensor, y_test_tensor
        )

        # Plot confusion matrix
        plot_confusion_matrix(confusion_matrix)

        logger.info("✅ Model evaluation completed successfully.")
        logger.info("Model accuracy: %.4f", accuracy)
    except Exception as e:
        logger.error("❌ Model Evaluation failed", exc_info=True)
        raise RuntimeError("evaluation failed") from e


if __name__ == "__main__":
    main()
