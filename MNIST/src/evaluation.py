# src/evaluation.py

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    predictions = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    acc = accuracy_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)

    print(f"✅ Accuracy: {acc:.4f}")
    return acc, cm

def plot_confusion_matrix(cm, labels=None, title="Confusion Matrix", save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if save_path:
        plt.savefig(save_path)
    plt.show()