import os
import yaml
import logging
import wandb
import numpy as np
from dotenv import load_dotenv

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.utils.logging_config import setup_logging
from src.data_load.data_loader import load_data
from src.data_validation.validation import validate_data
from src.data_preprocess.data_preprocessing import preprocess_data, save_preprocessed_data
from src.model.model import CNNModel, train_model, save_model
from src.evaluation.evaluation import evaluate_model, plot_confusion_matrix
from src.features.features import extract_embeddings, tsne_plot, pca_plot

PREPROCOUTPUT_TRAIN = 'data/processed/train/'
PREPROCOUTPUT_TEST = 'data/processed/test/'

# Load environment and login
load_dotenv()
wandb.login(key=os.getenv("wandb_api_key"))

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    setup_logging()
    config = load_config()

    wandb.init(
        project="mlops_mnist_project",
        config=config["model"],
        tags=["group2", "mnist", "baseline"],
        name="baseline_run"
    )

    # Load and preprocess
    (x_train, y_train), (x_test, y_test) = load_data()
    validate_data(x_train, y_train)
    validate_data(x_test, y_test)

    pp_x_train, pp_y_train = preprocess_data(x_train, y_train)
    save_preprocessed_data(pp_x_train, pp_y_train, PREPROCOUTPUT_TRAIN, "train")

    pp_x_test, pp_y_test = preprocess_data(x_test, y_test)
    save_preprocessed_data(pp_x_test, pp_y_test, PREPROCOUTPUT_TEST, "test")

    # Convert to PyTorch tensors
    x_train_tensor = torch.tensor(pp_x_train, dtype=torch.float32).permute(0, 3, 1, 2)
    y_train_tensor = torch.tensor(np.argmax(pp_y_train, axis=1), dtype=torch.long)
    x_test_tensor = torch.tensor(pp_x_test, dtype=torch.float32).permute(0, 3, 1, 2)
    y_test_tensor = torch.tensor(np.argmax(pp_y_test, axis=1), dtype=torch.long)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"], shuffle=True)

    # Train
    model = CNNModel(num_classes=config["model"]["num_classes"])
    model = train_model(model, train_loader, config)

    # Evaluate + plots
    acc, cm = evaluate_model(model, x_test_tensor, y_test_tensor)
    plot_confusion_matrix(cm, save_path="reports/figures/confusion_matrix.png")

    embeddings = extract_embeddings(model, x_test_tensor[:1000])
    labels_subset = y_test_tensor[:1000].numpy()
    np.savez("reports/embeddings/embeddings.npz", embeddings=embeddings, labels=labels_subset)

    # Limit to 500 points for faster rendering
    tsne_plot(embeddings[:500], labels_subset[:500])
    pca_plot(embeddings, labels_subset)

    # Save model
    save_path = config["model"]["save_path"]
    save_model(model, save_path)

    #Log single model artifact to W&B
    model_artifact = wandb.Artifact("pytorch_mnist_model", type="model")
    model_artifact.add_file(save_path)
    wandb.log_artifact(model_artifact)

    wandb.finish()

if __name__ == "__main__":
    main()
