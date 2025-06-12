import os
import sys
import logging
import argparse
import numpy as np

# Make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.model import load_model
from src.evaluation.evaluation import evaluate_model, plot_confusion_matrix
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="../../models/model.pth")
    parser.add_argument("--test-images-path", type=str, default="../../data/processed/test_images.npy")
    parser.add_argument("--test-labels-path", type=str, default="../../data/processed/test_labels.npy")
    args = parser.parse_args()

    logger.info("🔍 Starting evaluation step...")

    # Load model and data
    model = load_model(args.model_path)
    x_test = torch.tensor(np.load(args.test_images_path)).permute(0, 3, 1, 2).float()
    y_test = torch.tensor(np.argmax(np.load(args.test_labels_path), axis=1))

    acc, cm = evaluate_model(model, x_test, y_test)

    plot_confusion_matrix(cm, save_path="../../reports/figures/confusion_matrix.png")
    logger.info("✅ Evaluation finished. Accuracy = %.4f", acc)


if __name__ == "__main__":
    main()