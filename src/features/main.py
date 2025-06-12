import argparse
import logging
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.model.model import load_model
from src.features.features import extract_embeddings, tsne_plot, pca_plot


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="../../models/model.pth")
    parser.add_argument("--test-images-path", type=str, default="../../data/processed/test_images.npy")
    parser.add_argument("--test-labels-path", type=str, default="../../data/processed/test_labels.npy")
    parser.add_argument("--output-dir", type=str, default="../../reports/embeddings/")
    args = parser.parse_args()

    logger.info("🔎 Extracting and visualizing feature embeddings...")

    # Load model and test data
    model = load_model(args.model_path)
    x_test = np.load(args.test_images_path)
    y_test = np.load(args.test_labels_path)

    # Preprocess for torch input
    x_tensor = torch.tensor(x_test[:1000], dtype=torch.float32).permute(0, 3, 1, 2)
    y_tensor = torch.tensor(np.argmax(y_test[:1000], axis=1), dtype=torch.long)

    # Extract embeddings
    embeddings = extract_embeddings(model, x_tensor)

    # Save embeddings
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(
        os.path.join(args.output_dir, "embeddings.npz"),
        embeddings=embeddings,
        labels=y_tensor.numpy()
    )

    # Plot
    tsne_plot(embeddings, y_tensor.numpy(), save_path=os.path.join(args.output_dir, "tsne_plot.png"))
    pca_plot(embeddings, y_tensor.numpy(), save_path=os.path.join(args.output_dir, "pca_plot.png"))

    logger.info("✅ Feature extraction and visualization complete.")


if __name__ == "__main__":
    main()
