import hydra
from omegaconf import DictConfig
from src.utils.logging_config import get_logger
from src.data_load.data_loader import load_data
from src.data_preprocess.data_preprocessing import preprocess_data, save_preprocessed_data
from src.data_validation.validation import validate_data
import os

logger = get_logger(__name__)

@hydra.main(config_path='../../', config_name='config')
def run(cfg: DictConfig):
    logger.info("🧼 Starting data preprocessing step")

    # Load and validate
    (x_train, y_train), (x_test, y_test) = load_data()
    validate_data(x_train, y_train)
    validate_data(x_test, y_test)

    # Preprocess
    pp_x_train, pp_y_train = preprocess_data(x_train, y_train)
    pp_x_test, pp_y_test = preprocess_data(x_test, y_test)

    # Save
    os.makedirs(cfg.data_preprocess.output_dir, exist_ok=True)
    save_preprocessed_data(pp_x_train, pp_y_train, cfg.data_preprocess.output_dir, "train")
    save_preprocessed_data(pp_x_test, pp_y_test, cfg.data_preprocess.output_dir, "test")

    logger.info("✅ Preprocessing complete and data saved.")


if __name__ == "__main__":
    run()
