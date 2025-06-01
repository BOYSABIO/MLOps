import os
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
 
def evaluate_model(model, x_test, y_test):
    logging.info("Evaluating model...")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    with torch.no_grad():
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        outputs = model(x_test)
        _, predicted = torch.max(outputs, 1)

        y_true = y_test.cpu().numpy()
        y_pred = predicted.cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        logging.info(f"Evaluation complete - Accuracy: {acc:.4f}")
        return acc, cm
    
def plot_confusion_matrix(cm, labels=None, title="Confusion Matrix", save_path="reports/figures/confusion_matrix.png"):
    """
    Plot & save a confusion matrix heatmap to reports/figures/confusion_matrix.png

    Args:
        cm: Confusion matrix (2D array)
        labels: List of label names
        title: Plot title
        save_path: File path where figure will be saved (default: reports/figures/confusion_matrix.png)
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.figure(figsize=(8, 6))
        labels = labels if labels else list(range(cm.shape[0]))

        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
    
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Confusion Matrix Plot saved to {save_path}")
    except Exception as e:
        logging.error("Failed to save confusion matrix plot", exc_info=True)
        raise
