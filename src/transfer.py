"""Transfer-learning utility: freeze early layers and fine-tune the head on a small slice."""

from __future__ import annotations

import numpy as np
from tensorflow import keras


def fine_tune(model: keras.Model, X, Y, *, freeze_until: int, lr: float = 1e-5,
              epochs: int = 30, batch_size: int = 100, verbose: int = 2) -> keras.callbacks.History:
    """Freeze layers up to `freeze_until` (exclusive), then continue training on (X, Y).

    Pass freeze_until = len(model.layers) - 2 to fine-tune only the final dense head.
    """
    for layer in model.layers[:freeze_until]:
        layer.trainable = False
    for layer in model.layers[freeze_until:]:
        layer.trainable = True

    model.compile(optimizer=keras.optimizers.RMSprop(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model.fit(X, Y, epochs=epochs, batch_size=batch_size,
                     validation_split=0.1, verbose=verbose)


def random_subset(X: np.ndarray, Y: np.ndarray, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=n, replace=False)
    return X[idx], Y[idx]
