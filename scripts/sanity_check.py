import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def make_tuned(input_shape=(28, 28, 1), num_classes=10):
    inputs = keras.Input(shape=input_shape)
    x = layers.RandomRotation(0.06)(inputs)
    x = layers.RandomTranslation(0.07, 0.07)(x)
    x = layers.RandomZoom(0.06)(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("tensorflow", tf.__version__)
    model = make_tuned()
    x = np.zeros((1, 28, 28, 1), dtype=np.float32)
    y = model.predict(x)
    print("output shape", y.shape)
    assert y.shape == (1, 10)


if __name__ == "__main__":
    main()
