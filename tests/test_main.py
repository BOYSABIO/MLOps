import pytest
from unittest import mock
from omegaconf import OmegaConf
import src.main as main_module
import os

@pytest.fixture
def hydra_cfg():
    # Minimal valid config for the pipeline
    return OmegaConf.create({
        "step": "all",
        "paths": {
            "raw_data": "data/raw",
            "processed_data": "data/processed",
            "model": "models/model.pth",
            "reports": "reports/embeddings",
            "predictions": "predictions/prediction.txt"
        },
        "model": {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.001,
            "input_shape": [1, 28, 28],
            "num_classes": 10,
            "val_split": 0.2
        },
        "wandb": {
            "project": "mlops_mnist_project",
            "entity": "test-entity",
            "tags": ["test"],
            "name_prefix": "test",
            "enabled": False
        }
    })

def test_main_runs_all_steps(monkeypatch, hydra_cfg):
    # Patch mlflow.run to just record calls
    with mock.patch("mlflow.run") as mock_mlflow_run:
        main_module.main(hydra_cfg)
        # Should call mlflow.run for each step
        assert mock_mlflow_run.call_count >= 1

def test_main_invalid_step_logs_error(monkeypatch, hydra_cfg):
    hydra_cfg.step = "not_a_real_step"
    with mock.patch.object(main_module.logger, "error") as mock_log_error:
        main_module.main(hydra_cfg)
        mock_log_error.assert_called()
        assert "Unknown step" in mock_log_error.call_args[0][0]

def test_main_runs_single_step(monkeypatch, hydra_cfg):
    hydra_cfg.step = "model"
    with mock.patch("mlflow.run") as mock_mlflow_run:
        main_module.main(hydra_cfg)
        # Should call mlflow.run at least once for the model step
        assert mock_mlflow_run.call_count == 1
        assert "model" in mock_mlflow_run.call_args[1]["uri"]
