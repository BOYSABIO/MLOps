import os
import sys
import click
import logging

# Add the src directory to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from src.data_preprocess.data_preprocessing import preprocess_data, save_preprocessed_data
from src.utils.logging_config import get_logger

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

@click.command()
@click.option("--input-path", required=True, help="Path where the raw .npy files are located")
@click.option("--output-path", required=True, help="Path where the processed .npy files will be saved")
def main(input_path, output_path):
    """
    MLflow entry point para preprocesar los archivos .npy
    """
    try:
        logger.info("Step: Data Preprocessing executed.")
        logger.info(f"Processing data from: {input_path}")
        logger.info(f"Saving processed data to: {output_path}")

        # Load raw data
        import numpy as np
        
        x_train = np.load(os.path.join(input_path, "x_train.npy"))
        y_train = np.load(os.path.join(input_path, "y_train.npy"))
        x_test = np.load(os.path.join(input_path, "x_test.npy"))
        y_test = np.load(os.path.join(input_path, "y_test.npy"))
        
        # Preprocess data
        x_train_processed, y_train_processed = preprocess_data(x_train, y_train)
        x_test_processed, y_test_processed = preprocess_data(x_test, y_test)
        
        # Save processed data
        save_preprocessed_data(x_train_processed, y_train_processed, 
                             output_path, "train")
        save_preprocessed_data(x_test_processed, y_test_processed, 
                             output_path, "test")

        logger.info("✅ Data preprocessing completed successfully.")
    except Exception as e:
        logger.error("❌ Data Preprocessing failed", exc_info=True)
        raise RuntimeError("data_preprocess failed") from e

if __name__ == "__main__":
    main()