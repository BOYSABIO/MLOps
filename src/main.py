# main.py

from src.data_load.data_loader import load_data
#from src.evaluation.evaluation import evaluation
from src.utils.logging_config import setup_logging
import yaml
from src.data_preprocess.data_preprocessing import preprocess_data, save_preprocessed_data
from src.data_validation.validation import validate_data
from src.features.features import flatten_images
from src.model.model import build_model, compile_model, train_model

PREPROCOUTPUT_TRAIN = 'data/processed/train/'
PREPROCOUTPUT_TEST = 'data/processed/test/'

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    setup_logging()
    config = load_config()

    # Load data (custom loader)
    (x_train, y_train), (x_test, y_test) = load_data()

    # Data Validation
    validate_data(x_train, y_train)
    validate_data(x_test, y_test)

    # Preprocessing
    pp_x_train, pp_y_train = preprocess_data(x_train, y_train)
    save_preprocessed_data(pp_x_train, pp_y_train, PREPROCOUTPUT_TRAIN, "train")

    pp_x_test, pp_y_test = preprocess_data(x_test, y_test)
    save_preprocessed_data(pp_x_test, pp_y_test, PREPROCOUTPUT_TEST, "test")
    
    #data feature
    p_x_train = flatten_images(pp_x_train)
    p_x_test = flatten_images(pp_x_test)

    # # Model pipeline
    model = build_model(input_shape=tuple(config["model"]["input_shape"]),
                        num_classes=config["model"]["num_classes"])
    model = compile_model(model)
    model = train_model(model, pp_x_train, pp_y_train, config)

    # # Evaluation
    # acc, cm = evaluation.evaluate_model(m, x_test, y_test)
    # evaluation.plot_confusion_matrix(cm)

if __name__ == "__main__":
    main()