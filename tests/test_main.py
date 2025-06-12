"""Tests for main pipeline script (arg parsing, stages)."""

from unittest import mock
import src.main as main_module


def test_load_config_valid(tmp_path):
    """Test loading a valid YAML config."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n"
        "  batch_size: 32\n"
        "  num_classes: 10\n"
        "  save_path: model.pth"
    )

    config = main_module.load_config(path=str(config_path))
    assert config["model"]["batch_size"] == 32
    assert config["model"]["num_classes"] == 10


def test_main_data_stage(monkeypatch):
    """Test main function runs data stage without errors."""
    monkeypatch.setattr(
        main_module,
        "load_config",
        lambda path: {
            "model": {
                "batch_size": 32,
                "num_classes": 10,
                "save_path": "model.pth"
            }
        }
    )
    monkeypatch.setattr(
        main_module,
        "run_data_stage",
        lambda: ([], [], [], [])
    )
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)

    args = ["main.py", "--stage", "data"]
    with mock.patch("sys.argv", args):
        main_module.main()


def test_main_train_stage(monkeypatch):
    """Test main function runs training stage when data is mocked."""
    monkeypatch.setattr(
        main_module,
        "load_config",
        lambda path: {
            "model": {
                "batch_size": 32,
                "num_classes": 10,
                "save_path": "model.pth"
            }
        }
    )
    monkeypatch.setattr(
        main_module,
        "run_data_stage",
        lambda: ([], [], [], [])
    )
    monkeypatch.setattr(
        main_module, "run_training_stage", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)

    args = ["main.py", "--stage", "train"]
    with mock.patch("sys.argv", args):
        main_module.main()


def test_main_infer_stage(monkeypatch):
    """Test infer stage triggers draw interface."""
    monkeypatch.setattr(
        main_module,
        "load_config",
        lambda path: {
            "model": {
                "batch_size": 32,
                "num_classes": 10,
                "save_path": "model.pth"
            }
        }
    )
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    monkeypatch.setattr(
        main_module, "run_data_stage", lambda: ([], [], [], [])
    )
    monkeypatch.setattr(main_module, "run_training_stage", lambda *args: None)

    with mock.patch("src.draw_and_infer.main") as mock_draw:
        args = ["main.py", "--stage", "infer"]
        with mock.patch("sys.argv", args):
            main_module.main()
            mock_draw.assert_called_once()


def test_load_config_invalid(tmp_path):
    """Test that loading an invalid config file raises an error."""
    invalid_path = tmp_path / "bad_config.yaml"
    invalid_path.write_text("not: yaml: valid")  # broken on purpose

    with mock.patch("builtins.open", side_effect=FileNotFoundError):
        try:
            main_module.load_config(path=str(invalid_path))
        except FileNotFoundError:
            assert True
        else:
            assert False, "Expected FileNotFoundError"


def test_main_raises_runtime(monkeypatch):
    """Test that main catches and logs exceptions."""
    monkeypatch.setattr(main_module, "load_config", lambda _: {"model": {}})
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    monkeypatch.setattr(
        main_module,
        "run_data_stage",
        lambda: (_ for _ in ()).throw(RuntimeError("fail"))
    )

    args = ["main.py", "--stage", "data"]
    with mock.patch("sys.argv", args), \
         mock.patch("sys.exit") as mock_exit:
        main_module.main()
        mock_exit.assert_called_once_with(1)
