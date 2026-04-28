"""Keras model factories: logistic regression, MLP, CNN.

MLP and CNN follow the reference architectures from Prouff et al. (ASCAD paper).
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_logreg(input_len: int, n_classes: int) -> keras.Model:
    inp = keras.Input(shape=(input_len,))
    out = layers.Dense(n_classes, activation="softmax")(inp)
    m = keras.Model(inp, out, name="logreg")
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def build_mlp(input_len: int, n_classes: int, hidden: int = 200, depth: int = 6) -> keras.Model:
    inp = keras.Input(shape=(input_len,))
    x = inp
    for _ in range(depth):
        x = layers.Dense(hidden, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    m = keras.Model(inp, out, name="mlp")
    m.compile(optimizer=keras.optimizers.RMSprop(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def build_cnn(input_len: int, n_classes: int) -> keras.Model:
    inp = keras.Input(shape=(input_len, 1))
    x = inp
    filters = [64, 128, 256, 512, 512]
    for f in filters:
        x = layers.Conv1D(f, kernel_size=11, activation="relu", padding="same")(x)
        x = layers.AveragePooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dense(4096, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    m = keras.Model(inp, out, name="cnn")
    m.compile(optimizer=keras.optimizers.RMSprop(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m
