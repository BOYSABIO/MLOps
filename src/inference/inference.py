# src/inference/inference.py

import os
import torch
import numpy as np
import pandas as pd
import logging
from torchvision import transforms
from PIL import Image
from src.model.model import CNNModel
from src.data_preprocess.data_preprocessing import normalize_images

logger = logging.getLogger(__name__)

def load_model(model_path, device):
    model = CNNModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {model_path}")
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # grayscale
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)  # add batch dim

def run_inference(input_csv, config, output_csv):
    """
    Runs inference on a CSV listing paths to images.

    Args:
        input_csv: CSV with a column 'image_path' pointing to images.
        config: dict loaded from YAML.
        output_csv: path to save the predictions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(config["model"]["save_path"], device)

    # Load input paths
    df = pd.read_csv(input_csv)
    if "image_path" not in df.columns:
        raise ValueError("CSV must contain 'image_path' column")

    predictions = []
    for path in df["image_path"]:
        try:
            input_tensor = preprocess_image(path).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output, dim=1).item()
            predictions.append(pred)
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            predictions.append(None)

    df["prediction"] = predictions
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Inference complete. Results saved to {output_csv}")
