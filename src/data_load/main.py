import os
import sys
import click

# Add the src directory to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from src.data_load.data_loader import load_data, save_raw_data
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

@click.command()
@click.option("--output-path", required=True, help="Path where the .npy are saved")
def main(output_path):
    """
    MLflow entry point para cargar MNIST y guardar como archivos .npy
    """
    try:
        logger.info("Step: Data Load executed.")
        logger.info(f"Saving data in: {output_path}")

        (x_train, y_train), (x_test, y_test) = load_data(output_path)
        save_raw_data(x_train, y_train, x_test, y_test, output_path)

        logger.info("✅ .npy files saved successfuly.")
    except Exception as e:
        logger.error("❌ Data Load failed", exc_info=True)
        raise RuntimeError("data_load failed") from e

if __name__ == "__main__":
    main()