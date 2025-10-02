
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