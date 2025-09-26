from tensorflow import keras
import tensorflow as tf
import numpy as np
import time
import os

import tensorflow as tf
import numpy as np

def train_base(model, train_dataset, test_dataset, epochs, seed=None):
    """
    Train a model using standard supervised training with model.fit.
    """
    if seed:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    # Compile the model with loss + metrics
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    # Train using fit(), for PGD will need to do a custom training loop I think??
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        verbose=1
    )
    return model