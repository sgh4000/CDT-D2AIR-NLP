import pandas as pd
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

import os
import pickle as pk

import time

from character_perturbations import perturb_query

# Please note, that logic and code for embedding and rotating is as seen at: https://github.com/ANTONIONLP/ANTONIO/blob/main/src/data.py

def load_embeddings(dataset_name, encoding_model="all-MiniLM-L6-v2", encoding_model_name="all-MiniLM-L6-v2",
                    perturbation_name='original', n_perturbations=1, load_saved_embeddings=None, load_saved_align_mat=None,
                    data=None, path='datasets'):

    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]

    # Perturbations only to be applied to the training set
    # TODO - more perturbation types
    if perturbation_name == 'character':
        X_train = [perturb_query(x, n_perturbations) for x in X_train]

    # Embed
    encoder = SentenceTransformer(f'{encoding_model}')
    X_train = encoder.encode(X_train, show_progress_bar=False)
    X_test = encoder.encode(X_test, show_progress_bar=False)

    # Load shared alignment matrix (always from cleandata)
    align_mat = load_align_mat(dataset_name, encoding_model_name, X_train,
                               load_saved_align_mat, path=path, perturbation_name=perturbation_name)

    # Rotate
    X_train = np.matmul(X_train, align_mat)
    X_test = np.matmul(X_test, align_mat)

    # Save rotated embeddings + labels
    save_path = f'{path}/{dataset_name}/embeddings/{encoding_model_name}/{perturbation_name}'
    os.makedirs(save_path, exist_ok=True)
    np.save(f'{save_path}/X_train.npy', X_train)
    np.save(f'{save_path}/X_test.npy', X_test)
    np.save(f'{save_path}/y_train.npy', y_train)
    np.save(f'{save_path}/y_test.npy', y_test)

    # For now I'm returning the original text query for analysis
    return X_train, X_test, y_train, y_test, data[1]


def load_align_mat(dataset_name, encoding_model_name, data, load_saved_align_mat, path='datasets', perturbation_name='original'):
    # Always use the alignment from the ORIGINAL (clean) data to prevent duplication or drift issues
    align_mat_path = f'{path}/{dataset_name}/embeddings/{encoding_model_name}/original/align_mat.npy'

    if os.path.exists(align_mat_path):
        # Want to reuse where possible as alignment doesn't change with permutations
        align_mat = np.load(align_mat_path)
    else:
        # Only compute if this is clean data
        if perturbation_name != 'original':
            raise ValueError(
                "Alignment matrix must be computed on original data first.")

        u, s, vh = np.linalg.svd(a=data)
        align_mat = np.linalg.solve(a=vh, b=np.eye(len(data[0])))

        # Save align_mat to the original path
        save_path = os.path.dirname(align_mat_path)
        os.makedirs(save_path, exist_ok=True)
        np.save(align_mat_path, align_mat)

    return align_mat
