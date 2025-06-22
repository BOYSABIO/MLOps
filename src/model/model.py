import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def get_device():
    """
    Automatically select the appropriate device (MPS for Mac, CUDA for Nvidia,
    CPU fallback).
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class CNNModel(nn.Module):
    """
    A simple 3-layer CNN for grayscale image classification (e.g., MNIST).
    """
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def train_model(model, train_loader, config):
    """
    Trains a CNN model using CrossEntropyLoss and Adam optimizer,
    with optional wandb logging.

    Args:
        model: CNNModel instance
        train_loader: DataLoader for training data
        config: dict with 'model' keys: 'epochs', 'learning_rate'

    Returns:
        Trained model
    """
    try:
        # Try to import wandb for logging
        try:
            import wandb
            wandb_available = True
        except ImportError:
            wandb_available = False
            logger.info("WandB not available, continuing without logging")

        device = get_device()
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["model"].get("learning_rate", 1e-3)
        )
        epochs = config["model"]["epochs"]
        val_split = config["model"].get("val_split", 0.2)

        # Split train into train/val
        total_size = len(train_loader.dataset)
        val_size = int(val_split * total_size)
        train_size = total_size - val_size

        train_dataset, val_dataset = random_split(
            train_loader.dataset, [train_size, val_size]
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config["model"]["batch_size"], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config["model"]["batch_size"]
        )

        logger.info("Training on %s for %d epochs", device, epochs)
        model.train()

        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / len(train_loader)
            train_accuracy = correct / total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= len(val_loader)
            val_accuracy = val_correct / val_total

            # Log metrics
            if wandb_available and wandb.run is not None:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy
                }, step=epoch + 1)

            logger.info(
                "Epoch [%d/%d] | Train Loss: %.4f | Train Acc: %.4f | "
                "Val Loss: %.4f | Val Acc: %.4f",
                epoch + 1, epochs, train_loss, train_accuracy, 
                val_loss, val_accuracy
            )

        return model

    except Exception as e:
        logger.error("Model training failed", exc_info=True)
        raise RuntimeError("Training process failed") from e


def save_model(model, path):
    """
    Saves the model weights to disk.

    Args:
        model: Trained model
        path: File path for saving weights
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        logger.info("Model saved to %s", path)
        
        # Log model artifact to wandb if available
        try:
            import wandb
            if wandb.run is not None:
                artifact = wandb.Artifact("pytorch_mnist_model", type="model")
                artifact.add_file(path)
                wandb.log_artifact(artifact)
                logger.info("Model logged to WandB as artifact")
        except ImportError:
            pass  # WandB not available
            
    except Exception as e:
        logger.error("Saving model failed", exc_info=True)
        raise IOError("Failed to save model") from e


def load_model(path, num_classes=10):
    """
    Loads model weights from disk.

    Args:
        path: Path to .pt/.pth file
        num_classes: Output dimension of the model (default=10)

    Returns:
        A CNNModel instance with loaded weights
    """
    try:
        device = get_device()
        model = CNNModel(num_classes)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        logger.info("Model loaded from %s", path)
        return model
    except Exception as e:
        logger.error("Loading model failed", exc_info=True)
        raise RuntimeError("Failed to load model") from e
