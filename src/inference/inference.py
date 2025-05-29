import numpy as np
import tensorflow as tf
import logging
from utils.logging_config import setup_logging

logger = logging.getLogger()
setup_logging(log_file="../logs/inference_log.log")


def load_model(model_path):
    """Load and return the trained model."""
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def load_and_preprocess_data(data_path):
    """Load and preprocess new input data."""
    try:
        data = np.load(data_path)
        # Normalize data as needed (e.g., scale pixel values)
        data = data / 255.0
        logger.info(f"Data loaded and preprocessed from {data_path}, shape={data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading or preprocessing data: {e}")
        raise


def predict(model, data):
    """Run inference using the loaded model and preprocessed data."""
    try:
        predictions = model.predict(data)
        predicted_classes = np.argmax(predictions, axis=1)
        logger.info(f"Inference completed, total predictions made: {len(predicted_classes)}")
        return predicted_classes
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise


def save_predictions(predictions, output_path):
    """Save predictions to a file."""
    try:
        np.save(output_path, predictions)
        logger.info(f"Predictions saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
        raise


if __name__ == "__main__":
    import argparse
    import yaml

    # Load config
    with open("../config.yaml") as file:
        config = yaml.safe_load(file)

    parser = argparse.ArgumentParser(description="Run inference pipeline.")
    parser.add_argument("--model_path", default=config["model"]["path"])
    parser.add_argument("--data_path", default=config["inference"]["data_path"])
    parser.add_argument("--output_path", default=config["inference"]["output_path"])

    args = parser.parse_args()

    # Run pipeline
    model = load_model(args.model_path)
    data = load_and_preprocess_data(args.data_path)
    predictions = predict(model, data)
    save_predictions(predictions, args.output_path)
