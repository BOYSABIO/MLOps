import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logging_config import get_logger
from src.model.model import CNNModel, train_model, save_model
from src.data_load.data_loader import load_data
from src.data_validation.validation import validate_data
from src.data_preprocess.data_preprocessing import preprocess_data
from src.evaluation.evaluation import evaluate_model, plot_confusion_matrix
from src.features.features import extract_embeddings, tsne_plot, pca_plot

logger = get_logger(__name__)

@hydra.main(config_path='../../', config_name='config')
def run(cfg: DictConfig):
    logger.info("🚀 Starting model training step")

    # Load and preprocess
    (x_train, y_train), (x_test, y_test) = load_data()
    validate_data(x_train, y_train)
    validate_data(x_test, y_test)

    pp_x_train, pp_y_train = preprocess_data(x_train, y_train)
    pp_x_test, pp_y_test = preprocess_data(x_test, y_test)

    # Convert to tensors
    x_train_tensor = torch.tensor(pp_x_train, dtype=torch.float32).permute(0, 3, 1, 2)
    y_train_tensor = torch.tensor(np.argmax(pp_y_train, axis=1), dtype=torch.long)
    x_test_tensor = torch.tensor(pp_x_test, dtype=torch.float32).permute(0, 3, 1, 2)
    y_test_tensor = torch.tensor(np.argmax(pp_y_test, axis=1), dtype=torch.long)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=True
    )

    model = CNNModel(num_classes=cfg.model.num_classes)
    model = train_model(model, train_loader, cfg)

    _, cm = evaluate_model(model, x_test_tensor, y_test_tensor)
    plot_confusion_matrix(cm, save_path=cfg.evaluation.confusion_matrix_path)

    embeddings = extract_embeddings(model, x_test_tensor[:1000])
    labels_subset = y_test_tensor[:1000].numpy()

    np.savez(
        "reports/embeddings/embeddings.npz",
        embeddings=embeddings,
        labels=labels_subset
    )
    tsne_plot(embeddings, labels_subset)
    pca_plot(embeddings, labels_subset)

    save_model(model, cfg.model.save_path)

    logger.info("✅ Model training completed.")


if __name__ == "__main__":
    run()
