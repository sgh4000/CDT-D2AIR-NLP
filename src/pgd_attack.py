
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import random
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer

#only done on positive ones here but could do epsilon balls around all them?
def pgd_attack_embedded(model_base, X_pos_train_PCA, Y_pos_class_train):
    #picking values based on Katya
    epsilon = 0.05
    alpha = 0.01
    num_iter = 10
    #makes another copy here
    X_adv = tf.identity(X_pos_train_PCA)

    # Iterate PGD for num_iter steps
    for i in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(X_adv)  # Watch x_adv for gradient computation
            predictions = model_base(X_adv)  # Forward pass
            loss = tf.keras.losses.sparse_categorical_crossentropy(Y_pos_class_train, predictions)  # Loss w.r.t. true label

        # Compute the gradients of the loss w.r.t. the input
        gradients = tape.gradient(loss, X_adv)
        
        # Perform gradient ascent step in the direction that maximizes the loss
        perturbations = tf.sign(gradients)  # Use the sign of the gradients (FGSM-like step)
        X_adv = X_adv + alpha * perturbations  # Update the adversarial example
        
        # Project the adversarial example to ensure it's within epsilon-ball of the original image
        X_adv = tf.clip_by_value(X_adv, X_pos_train_PCA - epsilon, X_pos_train_PCA + epsilon)
        
        # Ensure the adversarial examples are within the valid input range [0, 1] commented out as only relevant for images
        #questions to consider: are embeddings scaled, standardised or left raw, and what are the numeric ranges of the PCA features?
        #X_adv = tf.clip_by_value(X_adv, 0.0, 1.0)
    
    return X_adv

#only done on positive ones here but could do epsilon balls around all them?
def pgd_attack_embedded_hyperrectangles(model, hyperrectangles, n_samples):
    #picking values based on Katya
    eps_multiplier = 1000
    pgd_steps = 5
    batch_size = 64

    pgd_dataset = []
    np.random.shuffle(hyperrectangles)
    for hyperrectangle in hyperrectangles[:n_samples]:
        t_hyperrectangle = np.transpose(hyperrectangle)

        # Calculate the epsilon for each dimension as ((dim[1] - dim[0]) / (pgd_steps * eps_multiplier))
        eps = []
        for d in hyperrectangle:
            eps.append((d[1] - d[0]) / (pgd_steps * eps_multiplier))
        
        # Generate a pgd point from the hyperrectangle 
        pgd_point = []
        for d in hyperrectangle:
            pgd_point.append(np.random.uniform(d[0], d[1]))
        # PGD attack on the image
        pgd_point = tf.convert_to_tensor([pgd_point], dtype=tf.float32)
        label_0 = tf.convert_to_tensor([[0]], dtype=tf.float32)
        for pgd_step in range(pgd_steps):
            with tf.GradientTape() as tape:
                tape.watch(pgd_point)
                prediction = model(pgd_point, training=False)  # Forward pass
                loss = tf.keras.losses.sparse_categorical_crossentropy(label_0, prediction) 
            # Get the gradients of the loss w.r.t to the input image.
            gradient = tape.gradient(loss, pgd_point)
            # Get the sign of the gradients to create the perturbation
            signed_grad = tf.sign(gradient)
            pgd_point = pgd_point + signed_grad * eps
            pgd_point = tf.clip_by_value(pgd_point, t_hyperrectangle[0], t_hyperrectangle[1])

        # Concatenate the pgd points
        if len(pgd_dataset) > 0:
            pgd_dataset = np.concatenate((pgd_dataset, pgd_point), axis=0)
        else:
            pgd_dataset = pgd_point

    pgd_dataset = np.asarray(pgd_dataset)
    pgd_labels_inside = np.full(len(pgd_dataset), 0)

    # Convert the pgd generated inputs into tf datasets, shuffle and batch them
    # pgd_dataset = tf.data.Dataset.from_tensor_slices((pgd_dataset, pgd_labels_inside))
    # pgd_dataset = pgd_dataset.shuffle(buffer_size=1024).batch(batch_size)

    return pgd_dataset, pgd_labels_inside