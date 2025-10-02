import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from data import pre_process, embed_and_align, embed_and_align_p, PCA_to_reduce_embeddings
from sentence_transformers import SentenceTransformer
from tensorflow import keras
import tensorflow as tf

def get_model():
    #good to keep model defined separately, initialiser seed and input_size specified in here, could take out to allow lots of runs
    input_size = 30
    initializer = tf.keras.initializers.GlorotUniform(seed=42)
    inputs = keras.Input(shape=(input_size,), name="embeddings")
    x = keras.layers.Dense(128, activation="relu", kernel_initializer=initializer, name="dense_1")(inputs)
    outputs = keras.layers.Dense(2, activation="linear", kernel_initializer=initializer, name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model

def train_base_model(X_pos_train_PCA, X_pos_test_PCA, X_neg_train_PCA, X_neg_test_PCA, Y_pos_class_train, Y_pos_class_test, Y_neg_class_train, Y_neg_class_test):
    #select what you would like:
    batch_size = 64
    epochs = 30
    
    #we want to combine the X data and the Y data - use concatenate so that 1D Y data doesn't get turned into column vectors which won't like later functions
    X_train = np.concatenate((X_pos_train_PCA, X_neg_train_PCA), axis=0)
    X_test = np.concatenate((X_pos_test_PCA, X_neg_test_PCA), axis=0)
    Y_train = np.concatenate((Y_pos_class_train, Y_neg_class_train), axis=0)
    Y_test = np.concatenate((Y_pos_class_test, Y_neg_class_test), axis=0)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

    #trains on batches and uses test dataset for validation data, 
    #not great practice maybe worth trying to implement k-folds especially if doing hyperparameter tuning
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy()

    model_base = get_model()
    model_base.compile(optimizer=optimizer, loss=loss_fn, metrics=[accuracy_fn])
    model_base.fit(train_dataset, epochs=epochs, validation_data=test_dataset,)

    return model_base, X_train, X_test, Y_train, Y_test, train_dataset, test_dataset