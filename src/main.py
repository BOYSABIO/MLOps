import hydra
from omegaconf import DictConfig
import mlflow
import os


@hydra.main(config_path="../", config_name="config")
def run_pipeline(cfg: DictConfig):
    # Make sure you're running from the root project directory
    os.chdir(hydra.utils.get_original_cwd())

    steps = cfg.get("steps")

    for step in steps:
        print(f"🔁 Running step: {step}")
        mlflow.run(f"src/{step}", parameters={}, use_conda=False)



if __name__ == "__main__":
    run_pipeline()
