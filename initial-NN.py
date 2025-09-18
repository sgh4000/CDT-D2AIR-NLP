import pandas as pd
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

# Load processed data set, of format [query,label], where 0 is non-medical and 1 is medical
df = pd.read_csv("medicheck-preprocessed.csv")

X = df["query"].tolist()
y = df["label"].values

print(f"Size of dataset: {len(X)}")
print(f"Medical(1)/non-medical(0) split: {pd.Series(y).value_counts().to_dict()}")

# S-BERT embedding on queries (Marcos's paper uses S-BERT so perhaps good place to start). Using pretrained model for this. https://github.com/Tgl70/DAIR-course-NLP/blob/main/src/train.py
# To-do: check with Jessica how they did embeddings? In their git repo I just see .npy embedding files for the data
# To-do: look into embedding normalisation? and definitely embedding gap

embedder = SentenceTransformer("all-MiniLM-L6-v2")
X_embeddings = embedder.encode(X, convert_to_numpy=True)


# Train/Test split, 80/20 seems sensible in both medical and non-medical classes
X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings, y, test_size=0.2, random_state=42, stratify=y
)

# Create NN model
input_size = X_embeddings.shape[1]
initializer = tf.keras.initializers.GlorotUniform(seed=42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_size,), name="input_features"),
    tf.keras.layers.Dense(128, activation="relu", kernel_initializer=initializer, name="dense_1"),
    tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer=initializer, name="output_layer")
])

# From a bit of Googling, Sigmoid + BinaryCrossentropy seems standard for [0,1] outputs, but needs more investigation

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
)

print(model.summary())

# To-do: early stopping? PGD attack/adversarial training definitely

# Training model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32,
    verbose=1
)

# Evaluate accuracy
loss, acc, auc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {acc:.3f}, AUC: {auc:.3f}")

# Trying out classification of some new questions to sanity check
check_questions = [
    "Which is the best painkiller for a headache?",
    "How do neural networks work?",
    "What are early signs of sepsis?",
    "How can I spend an afternoon in Edinburgh?"
]

new_embeddings = embedder.encode(check_questions, convert_to_numpy=True)
predictions_new = model.predict(new_embeddings)

print("\nPredictions of medical/not medical for new questions:")
for q, p in zip(check_questions, predictions_new):
    label = "Medical" if p > 0.5 else "Not medical"
    print(f"{q} → {label} (score={p[0]:.3f})")
