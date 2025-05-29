# main.py

from src.model.model import model
from src.data_load.data_loader import load_data
from src.evaluation.evaluation import evaluation
from src.utils.logging_config import setup_logging
import yaml
from keras.utils import to_categorical

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    setup_logging()
    config = load_config()

    # Load data (custom loader)
    (x_train, y_train), (x_test, y_test) = load_data()

    # Preprocessing
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Model pipeline
    m = model.build_model()
    m = model.compile_model(m)
    m = model.train_model(m, x_train, y_train)
    model.save_model(m)

    # Evaluation
    acc, cm = evaluation.evaluate_model(m, x_test, y_test)
    evaluation.plot_confusion_matrix(cm)