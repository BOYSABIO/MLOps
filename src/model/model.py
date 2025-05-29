import os
import logging
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, BatchNormalization, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

def build_model(input_shape=(28, 28, 1), num_classes=10) -> Model:
    """
    Build a CNN model for image classification.
    """
    logging.info(f"Building model with input shape {input_shape} and {num_classes} classes")
    inputs = Input(shape=input_shape)

    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def compile_model(model: Model) -> Model:
    """
    Compile the model with loss, optimizer, and metrics.
    """
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    logging.info("Model compiled successfully")
    return model

def train_model(model: Model, x_train: np.ndarray, y_train: np.ndarray, config: dict) -> Model:
    """
    Train the CNN model using parameters from config.
    """
    try:
        batch_size = config["model"]["batch_size"]
        epochs = config["model"]["epochs"]
        val_split = config.get("model", {}).get("val_split", 0.2)

        logging.info(f"Training model: batch_size={batch_size}, epochs={epochs}, val_split={val_split}")
        model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=val_split,
            verbose=2,
            callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
        )
        logging.info("Model training completed")
        return model
    except Exception as e:
        logging.error("Error during model training", exc_info=True)
        raise

def save_model(model: Model, path: str):
    """
    Save the model to the specified path.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model.save(path)
        logging.info(f"Model saved to {path}")
    except Exception as e:
        logging.error("Failed to save model", exc_info=True)
        raise

def load_existing_model(path: str) -> Model:
    """
    Load a model from a specified path.
    """
    try:
        model = load_model(path)
        logging.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logging.error("Failed to load model", exc_info=True)
        raise
