import pandas as pd
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

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

    # Want raw for future analysis that makes human sense
    X_train_pos_raw = data[0]
    X_train_neg_raw = data[1]
    X_test_pos_raw = data[2]
    X_test_neg_raw = data[3]
    y_train_pos = data[4]
    y_train_neg = data[5]
    y_test_pos = data[6]
    y_test_neg = data[7]

    # Perturbations only to be applied to the training set
    # TODO - more perturbation types
    if perturbation_name == 'character':
        X_train_pos_raw = [perturb_query(x, n_perturbations) for x in X_train_pos_raw]
        X_train_neg_raw = [perturb_query(x, n_perturbations) for x in X_train_neg_raw]

    # Embed
    encoder = SentenceTransformer(f'{encoding_model}')
    X_train_pos = encoder.encode(X_train_pos_raw, show_progress_bar=False)
    X_train_neg = encoder.encode(X_train_neg_raw, show_progress_bar=False)
    X_test_pos = encoder.encode(X_test_pos_raw, show_progress_bar=False)
    X_test_neg = encoder.encode(X_test_neg_raw, show_progress_bar=False)

    # Load shared alignment matrix (always from cleandata)
    align_mat = load_align_mat(dataset_name, encoding_model_name, X_train_pos,
                               load_saved_align_mat, path=path, perturbation_name=perturbation_name)

    # Rotate
    X_train_pos = np.matmul(X_train_pos, align_mat)
    X_train_neg = np.matmul(X_train_neg, align_mat)
    X_test_pos = np.matmul(X_test_pos, align_mat)
    X_test_neg = np.matmul(X_test_neg, align_mat)

    # Save rotated embeddings + labels
    save_path = f'{path}/{dataset_name}/embeddings/{encoding_model_name}/{perturbation_name}'
    os.makedirs(save_path, exist_ok=True)
    np.save(f'{save_path}/X_train_pos.npy', X_train_pos)
    np.save(f'{save_path}/X_train_neg.npy', X_train_neg)
    np.save(f'{save_path}/X_test_pos.npy', X_test_pos)
    np.save(f'{save_path}/X_test_neg.npy', X_test_neg)
    np.save(f'{save_path}/y_train_pos.npy', y_train_pos)
    np.save(f'{save_path}/y_train_neg.npy', y_train_neg)
    np.save(f'{save_path}/y_test_pos.npy', y_test_pos)
    np.save(f'{save_path}/y_test_neg.npy', y_test_neg)

    # For now I'm returning the original text query for analysis
    return X_train_pos, X_train_neg, X_test_pos, X_test_neg, y_train_pos, y_train_neg, y_test_pos, y_test_neg, X_test_pos_raw, X_test_neg_raw


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

def load_pca(dataset_name, encoding_model_name, load_saved_pca, X_train_pos, X_train_neg, X_test_pos, X_test_neg,  n_components=30, path='datasets'):
    if load_saved_pca:
        with open(f'{path}/{dataset_name}/embeddings/{encoding_model_name}/pca.pkl', 'rb') as pickle_file:
            data_pca = pk.load(pickle_file)

    else:
        # All data:
        data = np.vstack([X_train_pos, X_train_neg, X_test_pos, X_test_neg])
        # PCA data
        data_pca = PCA(n_components=n_components).fit(data)
        # Save the PCA
        save_path = f'{path}/{dataset_name}/embeddings/{encoding_model_name}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(f'{save_path}/pca.pkl', 'wb') as pickle_file:
            pk.dump(data_pca, pickle_file)

    X_train_pos = data_pca.transform(X_train_pos)
    X_train_neg = data_pca.transform(X_train_neg)
    X_test_pos = data_pca.transform(X_test_pos)
    X_test_neg = data_pca.transform(X_test_neg)

    # # Print the shape of the PCA data
    # print(f'Train pos sentence embeddings shape: {X_train_pos.shape}')
    # print(f'Train neg sentence embeddings shape: {X_train_neg.shape}')
    # print(f'Test pos sentence embeddings shape: {X_test_pos.shape}')
    # print(f'Test neg sentence embeddings shape: {X_test_neg.shape}')

    return X_train_pos, X_train_neg, X_test_pos, X_test_neg

