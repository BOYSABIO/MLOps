import hydra
from omegaconf import DictConfig
from src.utils.logging_config import get_logger
from src.data_load.data_loader import load_data
from src.data_validation.validation import validate_data

logger = get_logger(__name__)

@hydra.main(config_path='../../', config_name='config')
def run(cfg: DictConfig):
    logger.info("📥 Starting data load step")

    (x_train, y_train), (x_test, y_test) = load_data()
    
    validate_data(x_train, y_train)
    validate_data(x_test, y_test)

    logger.info(f"✅ Data loaded and validated — Train shape: {x_train.shape}, Test shape: {x_test.shape}")


if __name__ == "__main__":
    run()
