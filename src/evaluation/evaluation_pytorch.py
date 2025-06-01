import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import logging

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
