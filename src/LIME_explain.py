import numpy as np
import tensorflow as tf
from tensorflow import keras
from sentence_transformers import SentenceTransformer
from lime.lime_text import LimeTextExplainer
import webbrowser
import pickle as pk

# NOTE - please ensure you have installed LIME via - pip install LIME - https://github.com/marcotcr/lime

# Some quick LIME () overviews Local Interpretable Model-agnostic Explanations https://medium.com/nlplanet/two-minutes-nlp-explain-predictions-with-lime-aec46c7c25a2
# Class names
class_names = ["non-medical", "medical"]

def make_predict_fn(model, encoder, align_mat, pca):
    def predict_fn(texts):
        X = encoder.encode(texts, show_progress_bar=False)
        X = np.matmul(X, align_mat)
        X = pca.transform(X)
        preds = model.predict(X)
        return tf.nn.softmax(preds, axis=1).numpy()
    return predict_fn

def explain_text(query, model, encoder, align_mat, pca, class_names):
    predict_fn = make_predict_fn(model, encoder, align_mat, pca)
    # Doing text explainer, with re-embedding of the sample queries
    explainer = LimeTextExplainer(class_names=class_names, random_state=42)

    exp = explainer.explain_instance(
        query,
        predict_fn,
        num_features=10,
        labels=[0, 1]
    )

    print(f"\nLIME Explanation for: '{query}'")
    for label in [0, 1]:
        print(f"\nClass {label} ({class_names[label]}) explanation:")
        for word, weight in exp.as_list(label=label):
            print(f"  {word}: {weight:.4f}")

    print(exp.as_list())

    return exp


if __name__ == "__main__":
    # Load trained model - clean for now but TODO - permutation trained models
    model = keras.models.load_model("models/medical_query_clean.keras")
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    # Load encoder and use the alignment matrix already created
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    align_mat = np.load("datasets/medical_query_dataset/embeddings/all-MiniLM-L6-v2/align_mat.npy")
    
    with open("datasets/medical_query_dataset/embeddings/all-MiniLM-L6-v2/pca.pkl", "rb") as f:
        pca = pk.load(f)

    # Pick some queries - which I've taken from the test data set

    # With lime, I think the idea is that you get local explanations for individual query, TODO look at aggregation methods to get more robust view of whole model?

    queries = [
        "I need a endoscopy but no insurance and broke. Here is what Ive done so far",
        "Horrendous family history of cancer... How can I go about staying vigilant?",
        "LF Diggersby but will add anyone!",
        "Why is the community for this game nonexistent",
        "(Spoilers All) I've got to ask, why Stannis?",
        "Car dealership employees, how can the buyer get the best deal on vehicles?",
        "why am i feeling a strong pain in my left arm?",
        "how can i sort out my stomach problems? which medicines will make me feel better?"
        "i have been diagnosed with cancer",
        "cancer"
    ]

    for q in queries:
        explain_text(q, model, encoder, align_mat, pca, class_names)
