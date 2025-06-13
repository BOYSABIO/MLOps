# src/main.py

import argparse
import logging
import sys
import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from dotenv import load_dotenv
import wandb

from src.utils.logging_config import setup_logging
from src.data_load.data_loader import load_data
from src.data_validation.validation import validate_data
from src.data_preprocess.data_preprocessing import preprocess_data, save_preprocessed_data
from src.model.model import CNNModel, train_model, save_model
from src.evaluation.evaluation import evaluate_model, plot_confusion_matrix
from src.features.features import extract_embeddings, tsne_plot, pca_plot

# Constants
PREPROCOUTPUT_TRAIN = 'data/processed/train/'
PREPROCOUTPUT_TEST = 'data/processed/test/'
TSNE_LIMIT = 500  # For faster rendering

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_data_stage():
    logging.info("Running data stage...")
    (x_train, y_train), (x_test, y_test) = load_data()
    validate_data(x_train, y_train)
    validate_data(x_test, y_test)

    pp_x_train, pp_y_train = preprocess_data(x_train, y_train)
    pp_x_test, pp_y_test = preprocess_data(x_test, y_test)

    save_preprocessed_data(pp_x_train, pp_y_train, PREPROCOUTPUT_TRAIN, "train")
    save_preprocessed_data(pp_x_test, pp_y_test, PREPROCOUTPUT_TEST, "test")

    return pp_x_train, pp_y_train, pp_x_test, pp_y_test

def run_training_stage(config, pp_x_train, pp_y_train, pp_x_test, pp_y_test):
    logging.info("Running training stage...")

    # Convert to tensors
    x_train_tensor = torch.tensor(pp_x_train, dtype=torch.float32).permute(0, 3, 1, 2)
    y_train_tensor = torch.tensor(np.argmax(pp_y_train, axis=1), dtype=torch.long)
    x_test_tensor = torch.tensor(pp_x_test, dtype=torch.float32).permute(0, 3, 1, 2)
    y_test_tensor = torch.tensor(np.argmax(pp_y_test, axis=1), dtype=torch.long)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"], shuffle=True)

    model = CNNModel(num_classes=config["model"]["num_classes"])
    model = train_model(model, train_loader, config)

    # Evaluation
    _, cm = evaluate_model(model, x_test_tensor, y_test_tensor)
    plot_confusion_matrix(cm, save_path="reports/figures/confusion_matrix.png")

    embeddings = extract_embeddings(model, x_test_tensor[:1000])
    labels_subset = y_test_tensor[:1000].numpy()

    np.savez("reports/embeddings/embeddings.npz", embeddings=embeddings, labels=labels_subset)
    tsne_plot(embeddings[:TSNE_LIMIT], labels_subset[:TSNE_LIMIT])
    pca_plot(embeddings, labels_subset)

    # Save and log model
    save_path = config["model"]["save_path"]
    save_model(model, save_path)

    artifact = wandb.Artifact("pytorch_mnist_model", type="model")
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--stage", choices=["all", "data", "train", "infer"], default="all")
    args = parser.parse_args()

    setup_logging()
    logging.info("Pipeline started - stage: %s", args.stage)

    # Load config and environment
    config = load_config(args.config)
    load_dotenv()
    wandb.login(key=os.getenv("wandb_api_key"))

    wandb.init(
        project="mlops_mnist_project",
        config=config["model"],
        tags=["group2", "mnist", "baseline"],
        name="baseline_run"
    )

    pp_x_train = pp_y_train = pp_x_test = pp_y_test = None

    try:
        if args.stage in ("all", "data"):
            pp_x_train, pp_y_train, pp_x_test, pp_y_test = run_data_stage()

        if args.stage in ("all", "train"):
            if args.stage == "train" and pp_x_train is None:
                pp_x_train, pp_y_train, pp_x_test, pp_y_test = run_data_stage()
            run_training_stage(config, pp_x_train, pp_y_train, pp_x_test, pp_y_test)

        if args.stage == "infer":
            logging.info("Launching drawing interface for inference...")
            from src.draw_and_infer import main as draw_main
            draw_main()

    except (FileNotFoundError, ValueError, yaml.YAMLError, RuntimeError) as exc:
        logging.exception("Pipeline failed: %s", exc)
        sys.exit(1)

    wandb.finish()
    logging.info("Pipeline complete")

if __name__ == "__main__":
    main()
