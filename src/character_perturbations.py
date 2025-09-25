import random
import pandas as pd
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA
import os
import pickle as pk

# For future perturbation inspo: https://github.com/ANTONIONLP/ANTONIO/blob/main/src/perturbations.py

def delete_char(word):
    if len(word) > 1:
        idx = random.randrange(len(word))
        return word[:idx] + word[idx+1:]
    return word

def swap_chars(word):
    if len(word) > 1:
        idx = random.randrange(len(word) - 1)
        return word[:idx] + word[idx+1] + word[idx] + word[idx+2:]
    return word

def substitute_char(word):
    if len(word) > 0:
        idx = random.randrange(len(word))
        new_char = random.choice("abcdefghijklmnopqrstuvwxyz")
        return word[:idx] + new_char + word[idx+1:]
    return word

def perturb_query(query, n_perturbations=1):
    # Apply random perturbation(s) to random word(s) in the query
    words = query.split()
    for _ in range(n_perturbations):
        if not words:
            break
        # random word choice
        w_idx = random.randrange(len(words))
        word = words[w_idx]
        # random perturbation choice
        perturb_fn = random.choice([delete_char, swap_chars, substitute_char])
        words[w_idx] = perturb_fn(word)
    return " ".join(words)