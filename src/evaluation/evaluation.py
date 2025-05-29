# src/evaluation.py

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

def evaluate_model(model, x_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluate the model on test data and return accuracy and confusion matrix.

    Args:
        model: Trained model with a .predict() method
        x_test: Test data (images)
        y_test: True labels (one-hot encoded or integers)

    Returns:
        Tuple: (accuracy, confusion_matrix)
    """
    try:
        predictions = model.predict(x_test)
        predictions = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y_test, axis=1)

        acc = accuracy_score(true_labels, predictions)
        cm = confusion_matrix(true_labels, predictions)

        logging.info(f"Model evaluation completed. Accuracy {acc:.4f}")
        return acc, cm
    except Exception as e:
        logging.error("Error during model evaluation", exc_info = True)
    

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