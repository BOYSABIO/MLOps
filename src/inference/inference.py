import torch
import torch.nn.functional as F
import sys

# Add the app directory to Python path for imports
sys.path.append('/app')

from src.model.model import CNNModel

def load_trained_model(model_path="models/model.pth", device="cpu"):
    model = CNNModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_digits(model, image_tensor, device="cpu"):
    """
    Predicts labels for a batch of images.
    Args:
        model: Trained CNN model.
        image_tensor: Tensor of shape (N, 28, 28, 1)
        device: Device to run inference on.
    Returns:
        List of predicted labels.
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.permute(0, 3, 1, 2)  # Convert to (N, 1, 28, 28)
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        predictions = torch.argmax(F.softmax(output, dim=1), dim=1)
        return predictions.cpu().tolist()
