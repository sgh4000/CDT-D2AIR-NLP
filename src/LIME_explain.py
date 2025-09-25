import numpy as np
import tensorflow as tf
from tensorflow import keras
from sentence_transformers import SentenceTransformer
from lime.lime_text import LimeTextExplainer

# NOTE - please ensure you have installed LIME via - pip install LIME - https://github.com/marcotcr/lime

# Class names
class_names = ["non-medical", "medical"]

def make_predict_fn(model, encoder, align_mat):
    def predict_fn(texts):
        X = encoder.encode(texts, show_progress_bar=False)
        X = np.matmul(X, align_mat)
        preds = model.predict(X)
        return tf.nn.softmax(preds, axis=1).numpy()
    return predict_fn

def explain_text(query, model, encoder, align_mat, class_names):
    predict_fn = make_predict_fn(model, encoder, align_mat)
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

    # Pick some queries - which I've taken from the test data set
    queries = [
        "I need a endoscopy but no insurance and broke. Here is what Ive done so far",
        "Horrendous family history of cancer... How can I go about staying vigilant?",
        "LF Diggersby but will add anyone!",
        "Why is the community for this game nonexistent",
        "(Spoilers All) I've got to ask, why Stannis?",
        "Car dealership employees, how can the buyer get the best deal on vehicles?",
    ]

    for q in queries:
        explain_text(q, model, encoder, align_mat, class_names)
