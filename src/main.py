import mlflow
import os
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from dotenv import load_dotenv
from utils.logging_config import get_logger

logger = get_logger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger.info(f"Running pipeline step: {cfg.step}")

    # Load environment variables
    load_dotenv()

    VALID_STEPS = [
        "all", "data_load", "data_validation", "data_preprocess",
        "model", "evaluation", "features", "inference"
    ]
    if cfg.step not in VALID_STEPS:
        logger.error(
            f"Unknown step '{cfg.step}'. Must be one of: {VALID_STEPS}"
        )
        return

    # Use config-based paths with absolute path resolution
    raw_path = to_absolute_path(cfg.paths.raw_data)
    processed_path = to_absolute_path(cfg.paths.processed_data)
    model_path = to_absolute_path(cfg.paths.model)
    reports_path = to_absolute_path(cfg.paths.reports)
    predictions_path = to_absolute_path(cfg.paths.predictions)

    # Paths for NPY files
    train_images = os.path.join(processed_path, "train_images.npy")
    train_labels = os.path.join(processed_path, "train_labels.npy")
    test_images = os.path.join(processed_path, "test_images.npy")
    test_labels = os.path.join(processed_path, "test_labels.npy")

    try:
        # Launch modules with MLflow (keeping your existing setup)
        if cfg.step in ("all", "data_load"):
            logger.info("🔁 Running data_load step")
            mlflow.run(
                uri="src/data_load",
                entry_point="main",
                parameters={"output_path": raw_path},
            )

        if cfg.step in ("all", "data_validation"):
            logger.info("🔁 Running data_validation step")
            mlflow.run(
                uri="src/data_validation",
                entry_point="main",
                parameters={"input_path": raw_path},
            )

        if cfg.step in ("all", "data_preprocess"):
            logger.info("🔁 Running data_preprocess step")
            mlflow.run(
                uri="src/data_preprocess",
                entry_point="main",
                parameters={
                    "input_path": raw_path,
                    "output_path": processed_path
                },
            )

        if cfg.step in ("all", "model"):
            logger.info("🔁 Running model step")

            # Prepare wandb params to pass to the MLflow step
            wandb_params = {}
            if hasattr(cfg, 'wandb') and cfg.wandb.enabled:
                wandb_params = {
                    "wandb-project": cfg.wandb.project,
                    "wandb-entity": cfg.wandb.entity,
                    # Pass tags as a comma-separated string
                    "wandb-tags": ",".join(list(cfg.wandb.tags)),
                    "wandb-name-prefix": cfg.wandb.name_prefix,
                    "wandb-enabled": True
                }

            mlflow.run(
                uri="src/model",
                entry_point="main",
                parameters={
                    "train-images-path": train_images,
                    "train-labels-path": train_labels,
                    "output-model-path": model_path,
                    "epochs": cfg.model.epochs,
                    "learning-rate": cfg.model.learning_rate,
                    "batch-size": cfg.model.batch_size,
                    "val-split": cfg.model.val_split,
                    "num-classes": cfg.model.num_classes,
                    "input-shape": ",".join(map(str, cfg.model.input_shape)),
                    **wandb_params  # Add wandb parameters
                },
            )

        if cfg.step in ("all", "evaluation"):
            logger.info("🔁 Running evaluation step")
            mlflow.run(
                uri="src/evaluation",
                entry_point="main",
                parameters={
                    "model-path": model_path,
                    "test-images-path": test_images,
                    "test-labels-path": test_labels
                }
            )

        if cfg.step in ("all", "features"):
            logger.info("🔁 Running features step")
            mlflow.run(
                uri="src/features",
                entry_point="main",
                parameters={
                    "model-path": model_path,
                    "test-images-path": test_images,
                    "test-labels-path": test_labels,
                    "output-dir": reports_path
                }
            )

        if cfg.step in ("all", "inference"):
            logger.info("🔁 Running inference step")
            mlflow.run(
                uri="src/inference",
                entry_point="main",
                parameters={
                    "model-path": model_path,
                    "image-path": test_images,
                    "output-path": predictions_path
                }
            )

    finally:
        # No wandb.finish needed here anymore
        logger.info("MLOps pipeline finished.")


if __name__ == "__main__":
    main()
