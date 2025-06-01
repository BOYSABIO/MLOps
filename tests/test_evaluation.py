import pytest
import torch
from src.model.model_pytorch import CNNModel
from src.evaluation.evaluation_pytorch import evaluate_model

def test_evaluate_model_returns_metrics():
    model = CNNModel()
    x = torch.rand(32, 1, 28, 28)
    y = torch.randint(0, 10, (32,))
    
    acc, cm = evaluate_model(model, x, y)
    
    assert isinstance(acc, float)
    assert cm.shape == (10, 10)
