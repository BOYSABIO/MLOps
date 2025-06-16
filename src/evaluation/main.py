import hydra
from omegaconf import DictConfig
from src.evaluation.evaluation import evaluate_model

@hydra.main(config_path='../../', config_name='config')
def run(cfg: DictConfig):
    evaluate_model(cfg.evaluation)

if __name__ == "__main__":
    run()
