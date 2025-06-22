import click
import logging
from evaluation import evaluate_model_performance
from ..utils.logging_config import get_logger

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

@click.command()
@click.option("--model-path", required=True, help="Path to the trained model")
@click.option("--test-images-path", required=True, help="Path to test images")
@click.option("--test-labels-path", required=True, help="Path to test labels")
def main(model_path, test_images_path, test_labels_path):
    """
    MLflow entry point para evaluar el modelo
    """
    try:
        logger.info("Step: Model Evaluation executed.")
        logger.info(f"Evaluating model: {model_path}")
        logger.info(f"Test images: {test_images_path}")
        logger.info(f"Test labels: {test_labels_path}")

        evaluate_model_performance(model_path, test_images_path, test_labels_path)

        logger.info("✅ Model evaluation completed successfully.")
    except Exception as e:
        logger.error("❌ Model Evaluation failed", exc_info=True)
        raise RuntimeError("evaluation failed") from e

if __name__ == "__main__":
    main()