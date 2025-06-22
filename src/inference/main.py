import click
import logging
from inference import run_inference
from ..utils.logging_config import get_logger

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

        run_inference(model_path, image_path, output_path)

        logger.info("✅ Inference completed successfully.")
    except Exception as e:
        logger.error("❌ Inference failed", exc_info=True)
        raise RuntimeError("inference failed") from e

if __name__ == "__main__":
    main()
