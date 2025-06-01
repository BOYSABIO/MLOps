import logging
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.utils.logging_config import setup_logging
from src.data_load.data_loader import load_data
from src.data_validation.validation import validate_data
from src.data_preprocess.data_preprocessing import preprocess_data, save_preprocessed_data
from src.model.model import CNNModel, train_model, save_model
from src.evaluation.evaluation import evaluate_model, plot_confusion_matrix

PREPROCOUTPUT_TRAIN = 'data/processed/train/'
PREPROCOUTPUT_TEST = 'data/processed/test/'

def load_config(path="config2.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    setup_logging()
    logging.info("Starting PyTorch pipeline...")
    config = load_config()

    # Load raw data
    (x_train, y_train), (x_test, y_test) = load_data()

    # Data Validation
    validate_data(x_train, y_train)
    validate_data(x_test, y_test)


    # Preprocess
    pp_x_train, pp_y_train = preprocess_data(x_train, y_train)
    pp_x_test, pp_y_test = preprocess_data(x_test, y_test)

    save_preprocessed_data(pp_x_train, pp_y_train, PREPROCOUTPUT_TRAIN, "train")
    save_preprocessed_data(pp_x_test, pp_y_test, PREPROCOUTPUT_TEST, "test")

    # Fix shape: NHWC → NCHW
    x_train_tensor = torch.tensor(pp_x_train, dtype=torch.float32).permute(0, 3, 1, 2)
    y_train_tensor = torch.tensor(np.argmax(pp_y_train, axis=1), dtype=torch.long)

    x_test_tensor = torch.tensor(pp_x_test, dtype=torch.float32).permute(0, 3, 1, 2)
    y_test_tensor = torch.tensor(np.argmax(pp_y_test, axis=1), dtype=torch.long)


    # Create DataLoader
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"], shuffle=True)

    # Build + Train model
    model = CNNModel(num_classes=config["model"]["num_classes"])
    model = train_model(model, train_loader, config)

    # Evaluate
    acc, cm = evaluate_model(model, x_test_tensor, y_test_tensor)
    plot_confusion_matrix(cm, save_path="reports/figures/confusion_matrix.png")

    # Save model
    save_model(model, config["model"]["save_path"])

    logging.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
