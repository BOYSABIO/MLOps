import argparse
import logging
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import yaml

from src.model.model import CNNModel

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

 
def load_model(model_path, device):
    model = CNNModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {model_path}")
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension


def predict(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(F.softmax(output, dim=1), dim=1).item()
        return pred


def run_inference(image_folder, config_path, output_csv):
    setup_logging()

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_path = config["artifacts"]["model_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    # Predict on each image
    predictions = []
    image_dir = Path(image_folder)

    logger.info(f"Running inference on images in: {image_dir}")

    for image_file in image_dir.glob("*.png"):
        tensor = preprocess_image(image_file)
        label = predict(model, tensor, device)
        predictions.append({"filename": image_file.name, "prediction": label})

    # Save to CSV
    df = pd.DataFrame(predictions)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Predictions saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Run batch inference on image folder")
    parser.add_argument("image_folder", help="Folder containing image files (e.g., .png)")
    parser.add_argument("config_path", help="Path to config.yaml")
    parser.add_argument("output_csv", help="Path to save predictions CSV")
    args = parser.parse_args()

    run_inference(args.image_folder, args.config_path, args.output_csv)


if __name__ == "__main__":
    main()
