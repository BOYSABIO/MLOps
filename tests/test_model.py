import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model.model import CNNModel, train_model


def test_model_forward_pass():
    model = CNNModel()
    dummy_input = torch.randn(8, 1, 28, 28)
    output = model(dummy_input)
    assert output.shape == (8, 10)


def test_model_training_loop():
    x = torch.rand(16, 1, 28, 28)
    y = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    config = {"model": {"epochs": 1, "learning_rate": 1e-3}}
    model = CNNModel()
    trained = train_model(model, loader, config)

    assert isinstance(trained, CNNModel)
