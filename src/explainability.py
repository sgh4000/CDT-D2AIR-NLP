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
import random
import shap
import matplotlib.pyplot as plt
import numpy as np

#set seeds for reprodicibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

encoder = SentenceTransformer('all-MiniLM-L6-v2')

def make_predict_fn(align_matrix, pca, model_base):
    #predict_fn has to only take in the strings, but output probabilities
    #would be easier to save the matrices and models to locations and then could build directly without this nested function!
    def predict_fn(texts):
        embeds = encoder.encode(texts, show_progress_bar=False)
        embeds_align = np.matmul(embeds, align_matrix)
        embeds_PCA = pca.transform(embeds_align)
        logits = model_base.predict(embeds_PCA, verbose=0)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probs
    
    return predict_fn

def lime_test(X_pos_strings_test, predict_fn):

    explainer = LimeTextExplainer(class_names=['medical query', 'not medical query'])

    # Pick a random index from the test text array, not the PCA embeddings
    random_idx = random.randint(0, len(X_pos_strings_test) - 1)
    example_text = X_pos_strings_test[random_idx]  # raw string

    probs = predict_fn([example_text])[0]
    predicted_class = int(np.argmax(probs))

    exp = explainer.explain_instance(
        example_text,
        predict_fn,        # wrapped function that returns probabilities
        num_features=10,   # number of words to show in explanation
        top_labels=2
    )
    
    print(f"Random index: {random_idx}")
    print(f"Example text: {example_text}")
    print(f"Predicted class index: {predicted_class}")
    print(f"Class probabilities: {probs}")
    print(f"Mapped class: {explainer.class_names[predicted_class]}")

    #LIME explanation
    lime_list = exp.as_list(label=predicted_class)
    print("LIME explanation:", lime_list)

    # # Save interactive HTML - but rendering bug so show twice!
    exp.save_to_file(f"data/explainability/lime_text_index-{random_idx}.html")
    return random_idx



def SHAP_vis(predict_fn, Vis_X_test_string, random_idx):
    explainer = shap.Explainer(predict_fn, masker=shap.maskers.Text())
    shap_values = explainer([Vis_X_test_string[random_idx]])
    html_str = shap.plots.text(shap_values[0], display=False)
    with open(f"data/explainability/shap_text_index{random_idx}.html", "w") as f:
        f.write(html_str)