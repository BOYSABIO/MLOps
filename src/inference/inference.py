# src/inference/inference.py

import torch
import torch.nn.functional as F
from src.model.model import CNNModel

def load_trained_model(model_path="models/mnist_model.pth", device="cpu"):
    model = CNNModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_digit(model, image_tensor, device="cpu"):
    """
    Takes a preprocessed image tensor of shape (1, 1, 28, 28) and returns predicted digit.
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        return torch.argmax(F.softmax(output, dim=1), dim=1).item()
