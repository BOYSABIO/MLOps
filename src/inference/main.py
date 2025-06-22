import os
import sys
import click
import logging
import numpy as np
import torch

# Add the src directory to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from src.inference.inference import load_trained_model, predict_digits
from src.utils.logging_config import get_logger

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

@click.command()
@click.option("--model-path", required=True, help="Path to the trained model")
@click.option("--image-path", required=True, help="Path to input image")
@click.option("--output-path", required=True, help="Path to save prediction results")
def main(model_path, image_path, output_path):
    """
    MLflow entry point para realizar inferencia
    """
    try:
        logger.info("Step: Inference executed.")
        logger.info(f"Model: {model_path}")
        logger.info(f"Input image: {image_path}")
        logger.info(f"Output path: {output_path}")

        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_trained_model(model_path, device)
        
        # Load test images (using first 10 for inference demo)
        x_test = np.load(image_path)
        x_test_tensor = torch.tensor(x_test[:10], dtype=torch.float32)
        
        # Make predictions
        predictions = predict_digits(model, x_test_tensor, device)
        
        # Save predictions
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for i, pred in enumerate(predictions):
                f.write(f"Image {i}: Predicted digit = {pred}\n")
        
        logger.info(f"Predictions saved to {output_path}")
        logger.info(f"Sample predictions: {predictions[:5]}")

        logger.info("✅ Inference completed successfully.")
    except Exception as e:
        logger.error("❌ Inference failed", exc_info=True)
        raise RuntimeError("inference failed") from e

if __name__ == "__main__":
    main()
