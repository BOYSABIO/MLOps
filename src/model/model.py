import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
    Trains a CNN model using CrossEntropyLoss and Adam optimizer.

    Args:
        model: CNNModel instance
        train_loader: DataLoader for training data
        config: dict with 'model' keys: 'epochs', 'learning_rate'

    Returns:
        Trained model
    """
    try:
        device = get_device()
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["model"].get("learning_rate", 1e-3)
        )
        epochs = config["model"]["epochs"]

        logging.info("Training on %s for %d epochs", device, epochs)
        model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            logging.info(
                "Epoch [%d/%d], Loss: %.4f",
                epoch + 1, epochs, running_loss
            )

        return model

    except Exception as e:
        logging.error("Model training failed", exc_info=True)
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
        logging.info("Model saved to %s", path)
    except Exception as e:
        logging.error("Saving model failed", exc_info=True)
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
        logging.info("Model loaded from %s", path)
        return model
    except Exception as e:
        logging.error("Loading model failed", exc_info=True)
        raise RuntimeError("Failed to load model") from e
