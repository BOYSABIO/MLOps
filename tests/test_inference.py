import pytest
import numpy as np

from src.inference.inference import load_model, load_and_preprocess_data, predict

def test_load_model():
    model = load_model("./models/mnist-model.h5")
    assert model is not None

def test_load_and_preprocess_data():
    data = load_and_preprocess_data("./data/raw/x_test.npy")
    assert isinstance(data, np.ndarray)
    assert data.shape == (10000, 28, 28)
    assert data.max() <= 1.0

def test_predict():
    model = load_model("./models/mnist-model.h5")
    data = np.random.rand(5, 28, 28)
    predictions = predict(model, data)
    assert len(predictions) == 5
    assert predictions.dtype == np.int64
