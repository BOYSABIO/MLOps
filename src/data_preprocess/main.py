import click
import logging
from data_preprocessing import preprocess_and_save

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

        preprocess_and_save(input_path, output_path)

        logger.info("✅ Data preprocessing completed successfully.")
    except Exception as e:
        logger.error("❌ Data Preprocessing failed", exc_info=True)
        raise RuntimeError("data_preprocess failed") from e

if __name__ == "__main__":
    main()