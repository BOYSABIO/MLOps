import pytest
import numpy as np
import os
import shutil
from src import model
from src.evaluation import evaluate_model, plot_confusion_matrix
import matplotlib
matplotlib.use('Agg')

FIGURE_PATH = "reports/figures/confusion_matrix.png"

def test_evaluate_model_returns_accuracy_and_cm():
    m = model.build_model()
    m = model.compile_model(m)

    # Create dummy data
    x = np.random.rand(100, 28, 28)
    y = np.zeros((100, 10))
    y[np.arange(100), np.random.randint(0, 10, 100)] = 1

    m.fit(x, y, epochs=1, batch_size=10, verbose=0)

    acc, cm = evaluate_model(m, x, y)
    
    assert isinstance(acc, float), "Accuracy should be a float"
    assert cm.shape == (10, 10), "Confusion matrix should be 10x10"

def test_plot_confusion_matrix_saves_file():
    cm = np.random.randint(0, 20, size=(10, 10))

    # Ensure clean state
    if os.path.exists(FIGURE_PATH):
        os.remove(FIGURE_PATH)

    plot_confusion_matrix(cm, save_path=FIGURE_PATH)

    assert os.path.exists(FIGURE_PATH), "Confusion matrix plot was not saved"