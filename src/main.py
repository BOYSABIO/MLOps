# main.py

#from src.model.model import model
from src.data_load.data_loader import load_data
#from src.evaluation.evaluation import evaluation
from src.utils.logging_config import setup_logging
import yaml
from keras.utils import to_categorical
from src.data_preprocess.data_preprocessing import preprocess_data, save_preprocessed_data

PREPROCOUTPUT_TRAIN = 'data/processed/train/'
PREPROCOUTPUT_TEST = 'data/processed/test/'

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    setup_logging()
    config = load_config()

    # Load data (custom loader)
    (x_train, y_train), (x_test, y_test) = load_data()

    # Data Validation
    # ...

    # Preprocessing
    pp_x_train, pp_y_train = preprocess_data(x_train, y_train)
    save_preprocessed_data(pp_x_train, pp_y_train, PREPROCOUTPUT_TRAIN)

    pp_x_test, pp_y_test = preprocess_data(x_test, y_test)
    save_preprocessed_data(pp_x_test, pp_y_test, PREPROCOUTPUT_TEST)

    # Model pipeline
    m = model.build_model()
    m = model.compile_model(m)
    m = model.train_model(m, x_train, y_train)
    model.save_model(m)

    # Evaluation
    acc, cm = evaluation.evaluate_model(m, x_test, y_test)
    evaluation.plot_confusion_matrix(cm)