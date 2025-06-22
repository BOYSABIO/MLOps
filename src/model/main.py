import click
import logging
import numpy as np
import torch
import sys
import os

# Add src/ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.model.model import CNNModel, train_model, save_model


@click.command()
@click.option("--train-images-path", required=True, type=str)
@click.option("--train-labels-path", required=True, type=str)
@click.option("--output-model-path", required=True, type=str)
@click.option("--epochs", default=5, type=int)
@click.option("--learning-rate", default=0.001, type=float)
@click.option("--batch-size", default=32, type=int)
@click.option("--val-split", default=0.2, type=float)
def main(train_images_path, train_labels_path, output_model_path, epochs, learning_rate, batch_size, val_split):
    logging.basicConfig(level=logging.INFO)
    logging.info("✅ Model training started")

    x = np.load(train_images_path)
    y = np.load(train_labels_path)

    x_tensor = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2)
    y_tensor = torch.tensor(np.argmax(y, axis=1), dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CNNModel(num_classes=10)
    config = {
        "model": {
            "epochs": epochs, 
            "learning_rate": learning_rate, 
            "batch_size": batch_size,
            "val_split": val_split
        }
    }
    model = train_model(model, loader, config)

    save_model(model, output_model_path)
    logging.info("✅ Model training complete and saved")


if __name__ == "__main__":
    main()