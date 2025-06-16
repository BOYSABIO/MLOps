import hydra
from omegaconf import DictConfig
from src.features.features import build_features

@hydra.main(config_path='../../', config_name='config')
def run(cfg: DictConfig):
    build_features(cfg.features)

if __name__ == "__main__":
    run()
