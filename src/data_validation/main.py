import os
import click
import numpy as np
from validation import validate_data
import logging

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--input-path", type=click.Path(exists=True), required=True)
def main(input_path):
    logger.info(f"Step: data_validation started")
    logger.info(f"Validating data in {input_path}...")

    try:
        x_path = os.path.join(input_path, "x_train.npy")
        y_path = os.path.join(input_path, "y_train.npy")

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            raise FileNotFoundError("Expected .npy files not found in input path.")

        x = np.load(x_path)
        y = np.load(y_path)

        validate_data(x, y)

        logger.info("✅ Data validation passed.")

    except Exception as e:
        logger.error("❌ Data validation failed.", exc_info=True)
        raise e


if __name__ == "__main__":
    main()