"""MLflow entry point to perform inference using a trained model."""

import logging
import os
import click
import numpy as np
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.inference.inference import load_trained_model, predict_digits
from src.utils.logging_config import get_logger

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


@click.command()
@click.option("--model-path", required=True, help="Path to the trained model")
@click.option("--image-path", required=True, help="Path to input image")
@click.option("--output-path", required=True,
              help="Path to save prediction results")
def main(model_path, image_path, output_path):
    """
    MLflow entry point para realizar inferencia.
    """
    try:
        logger.info("Step: Inference executed.")
        logger.info("Model: %s", model_path)
        logger.info("Input image: %s", image_path)
        logger.info("Output path: %s", output_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_trained_model(model_path, device)

        x_test = np.load(image_path)
        x_test_tensor = torch.tensor(x_test[:10], dtype=torch.float32)

        predictions = predict_digits(model, x_test_tensor, device)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for i, pred in enumerate(predictions):
                f.write("Image %d: Predicted digit = %s\n" % (i, pred))

        logger.info("Predictions saved to %s", output_path)
        logger.info("Sample predictions: %s", predictions[:5])
        logger.info("✅ Inference completed successfully.")
    except Exception as e:
        logger.error("❌ Inference failed", exc_info=True)
        raise RuntimeError("inference failed") from e


if __name__ == "__main__":
    main()
