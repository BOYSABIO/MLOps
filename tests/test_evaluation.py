import pytest
from src import model, evaluation
import numpy as np

def test_evaluate_model_returns_accuracy_and_cm():
    m = model.build_model()
    m = model.compile_model(m)

    x = np.random.rand(100, 28, 28)
    y = np.zeros((100, 10))
    y[np.arange(100), np.random.randint(0, 10, 100)] = 1

    m.fit(x, y, epochs=1, batch_size=10, verbose=0)

    acc, cm = evaluation.evaluate_model(m, x, y)
    assert isinstance(acc, float), "Accuracy should be a float"
    assert cm.shape == (10, 10), "Confusion matrix should be 10x10"

def test_plot_confusion_matrix_executes():
    import matplotlib
    matplotlib.use('Agg')  # avoid GUI
    import numpy as np

    cm = np.random.randint(0, 10, size=(10, 10))
    evaluation.plot_confusion_matrix(cm)  # Just check it runs