import os
import logging
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate_model(model, x_test, y_test):
    """
    Evaluates the model on the given test dataset.

    Args:
        model: Trained PyTorch model.
        x_test: Input features (torch.Tensor).
        y_test: Ground truth labels (torch.Tensor).

    Returns:
        Tuple of (accuracy, confusion_matrix)
    """
    logging.info("Evaluating model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    try:
        with torch.no_grad():
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            outputs = model(x_test)
            if outputs.ndim == 1 or outputs.size(1) == 1:
                logging.warning("Model output may not be multi-class logits")

            _, predicted = torch.max(outputs, 1)

            y_true = y_test.cpu().numpy()
            y_pred = predicted.cpu().numpy()

            acc = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)

            logging.info("Evaluation complete - Accuracy: %.4f", acc)
            return acc, cm

    except Exception as e:
        logging.error("Model evaluation failed", exc_info=True)
        raise RuntimeError("Failed to evaluate model") from e


def plot_confusion_matrix(
    cm,
    labels=None,
    title="Confusion Matrix",
    save_path="../../reports/figures/confusion_matrix.png"
):
    """
    Plots and saves a confusion matrix heatmap.

    Args:
        cm: Confusion matrix (2D numpy array).
        labels: List of label names.
        title: Title for the heatmap.
        save_path: Where to save the image.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.figure(figsize=(8, 6))
        labels = labels if labels else list(range(cm.shape[0]))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Purples",
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info("Confusion matrix plot saved to %s", save_path)
    except Exception as e:
        logging.error("Failed to plot/save confusion matrix", exc_info=True)
        raise RuntimeError("Failed to generate confusion matrix plot") from e
