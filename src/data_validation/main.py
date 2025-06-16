import hydra
from omegaconf import DictConfig
from src.data_validation.validation import validate_data

@hydra.main(config_path='../../', config_name='config')
def run(cfg: DictConfig):
    validate_data(cfg.data_validation)

if __name__ == "__main__":
    run()
