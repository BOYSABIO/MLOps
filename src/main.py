import mlflow
import os
import hydra
import logging
from omegaconf import DictConfig
from hydra.utils import to_absolute_path


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging.info(f"Running pipeline step: {cfg.step}")

    VALID_STEPS = [
        "all", "data_load", "data_validation", "data_preprocess",
        "model", "evaluation", "features", "inference"
    ]
    if cfg.step not in VALID_STEPS:
        logging.error(
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

    # Launch modules with MLflow (keeping your existing setup)
    if cfg.step in ("all", "data_load"):
        mlflow.run(
            uri="src/data_load",
            entry_point="main",
            parameters={"output_path": raw_path},
        )

    if cfg.step in ("all", "data_validation"):
        mlflow.run(
            uri="src/data_validation",
            entry_point="main",
            parameters={"input_path": raw_path},
        )

    if cfg.step in ("all", "data_preprocess"):
        mlflow.run(
            uri="src/data_preprocess",
            entry_point="main",
            parameters={
                "input_path": raw_path,
                "output_path": processed_path
            },
        )

    if cfg.step in ("all", "model"):
        mlflow.run(
            uri="src/model",
            entry_point="main",
            parameters={
                "train-images-path": train_images,
                "train-labels-path": train_labels,
                "output-model-path": model_path,
                "epochs": cfg.model.epochs,
                "learning-rate": cfg.model.learning_rate,
                "batch-size": cfg.model.batch_size
            }
        )

    if cfg.step in ("all", "evaluation"):
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
        mlflow.run(
            uri="src/inference",
            entry_point="main",
            parameters={
                "model-path": model_path,
                "image-path": test_images,
                "output-path": predictions_path
            }
        )


if __name__ == "__main__":
    main()
