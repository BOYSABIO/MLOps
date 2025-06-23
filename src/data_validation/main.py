"""MLflow entry point to validate MNIST .npy files."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import logging

import click
import numpy as np

from src.data_validation.validation import validate_data
from src.utils.logging_config import get_logger

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


@click.command()
@click.option("--input-path", required=True,
              help="Path where the .npy files are located")
def main(input_path):
    """
    MLflow entry point para validar los archivos .npy.
    """
    try:
        logger.info("Step: Data Validation executed.")
        logger.info("Validating data in: %s", input_path)

        # Load and validate the data files
        x_train = np.load(os.path.join(input_path, "x_train.npy"))
        y_train = np.load(os.path.join(input_path, "y_train.npy"))
        x_test = np.load(os.path.join(input_path, "x_test.npy"))
        y_test = np.load(os.path.join(input_path, "y_test.npy"))

        # Validate each dataset
        validate_data(x_train, y_train)
        validate_data(x_test, y_test)

        logger.info("✅ Data validation completed successfully.")
    except Exception as e:
        logger.error("❌ Data Validation failed", exc_info=True)
        raise RuntimeError("data_validation failed") from e


if __name__ == "__main__":
    main()
