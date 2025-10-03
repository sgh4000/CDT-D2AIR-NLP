
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from data import pre_process, embed_and_align, embed_and_align_p, PCA_to_reduce_embeddings
from train import get_model, train_base_model
from metrics import print_metrics
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

encoder = SentenceTransformer('all-MiniLM-L6-v2')

def make_predict_fn(align_matrix, pca, model_base):
    """
    Returns a predict_fn that LIME can use, with all required dependencies baked in.
    """
    def predict_fn(texts):
        embeds = encoder.encode(texts, show_progress_bar=False)
        embeds_align = np.matmul(embeds, align_matrix)
        embeds_PCA = pca.transform(embeds_align)
        logits = model_base.predict(embeds_PCA)
        probs = tf.nn.softmax(logits, axis=1).numpy()
        return probs
    
    return predict_fn

def lime_test(X_pos_strings_test, predict_fn):

    explainer = LimeTextExplainer(class_names=['medical query', 'not medical query'])


    # 2. Pick a random index from X_base_test (or your test text array)
    # Pick a random index from the test text array, not the PCA embeddings
    random_idx = random.randint(0, len(X_pos_strings_test) - 1)
    example_text = X_pos_strings_test[random_idx]  # <-- raw string


    # 3. Generate explanation
    exp = explainer.explain_instance(
        example_text,
        predict_fn,        # your wrapped function that returns probabilities
        num_features=10,   # number of words to show in explanation
        top_labels=1       # only explain the top predicted class
    )

    probs = predict_fn([example_text])[0]
    predicted_class = np.argmax(probs)

    print(f"Random index: {random_idx}")
    print(f"Example text: {example_text}")
    print(f"Predicted class index: {predicted_class}")
    print(f"Class probabilities: {probs}")
    print(f"Mapped class: {explainer.class_names[predicted_class]}")

    # LIME explanation
    lime_list = exp.as_list(label=predicted_class)
    print("LIME explanation:", lime_list)

    # Save interactive HTML
    exp.save_to_file('data/explainability/oi.html')


    
    return


