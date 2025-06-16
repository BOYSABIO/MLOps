import hydra
from omegaconf import DictConfig
from src.inference.inference import run_inference

@hydra.main(config_path='../../', config_name='config')
def run(cfg: DictConfig):
    run_inference(cfg.inference)

if __name__ == "__main__":
    run()
