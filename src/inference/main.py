import argparse
import numpy as np
import torch
import os
import logging

from inference import load_trained_model, predict_digits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/model.pth")
    parser.add_argument("--image-path", type=str, default="data/processed/test_images.npy")
    parser.add_argument("--output-path", type=str, default="predictions/prediction.txt")
    args = parser.parse_args()

    logger.info("🧠 Starting inference...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(args.model_path, device=device)

    image = torch.from_numpy(np.load(args.image_path)).float()

    predictions = predict_digits(model, image, device=device)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        for p in predictions:
            f.write(f"{p}\n")

    logger.info(f"✅ Inference complete. Predictions saved to {args.output_path}")


if __name__ == "__main__":
    main()
