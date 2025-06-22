import click
import logging
from validation import validate_data_files
from ..utils.logging_config import get_logger

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

@click.command()
@click.option("--input-path", required=True, help="Path where the .npy files are located")
def main(input_path):
    """
    MLflow entry point para validar los archivos .npy
    """
    try:
        logger.info("Step: Data Validation executed.")
        logger.info(f"Validating data in: {input_path}")

        validate_data_files(input_path)

        logger.info("✅ Data validation completed successfully.")
    except Exception as e:
        logger.error("❌ Data Validation failed", exc_info=True)
        raise RuntimeError("data_validation failed") from e

if __name__ == "__main__":
    main()