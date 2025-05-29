# src/model.py

from keras.models import Model, load_model
from keras.layers import Input, Conv1D, BatchNormalization, Dropout, Flatten, Dense
from keras.optimizers import Adam
import os

def build_model(input_shape=(28, 28), num_classes=10):
    i_layer = Input(shape=input_shape)

    x = Conv1D(32, 3, strides=1, padding='same', activation='relu')(i_layer)
    x = BatchNormalization(momentum=0.9)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(32, 3, strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(32, 3, strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)

    o_layer = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=i_layer, outputs=o_layer)
    return model

def compile_model(model):
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, batch_size=1000, epochs=20, val_split=0.2):
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=val_split,
              verbose=2)
    return model

def save_model(model, path="models/mnist_model.h5"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)

def load_existing_model(path="models/mnist_model.h5"):
    return load_model(path)