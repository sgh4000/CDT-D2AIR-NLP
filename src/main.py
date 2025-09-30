import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
#function to preprocess the raw data before embeddings
def pre_process():
    #should read the data from the files, and begin extract the relevant strings and label them
    #do I need to include severity levels, and indexing here?
    #save them where needed?
    #do i need to do train testing split here?
    #THIS IS TAKEN FROM ANTONIO but sci-kit learn used to shuffle instead - may need to change this eventually
    expert = pd.read_csv("data/raw/medicheck-expert.csv")

    #check how this is handled
    med_neg = pd.read_csv("data/raw/medicheck-neg.csv")


    # Select the X and y columns, then the rows that belongs to classes 1 to 3, and finally replace the labes with only 1
    expert = expert[['query', 'query-label-expert']]
    pos = expert.loc[(expert['query-label-expert'] >= 1) & (expert['query-label-expert'] <= 3)]
    pos['query-label-expert'] = pos['query-label-expert'].replace({1: 0, 2: 0, 3: 0})

    neg = expert.loc[expert['query-label-expert'] == 0]
    neg['query-label-expert'] = neg['query-label-expert'].replace({0: 1})

    train_pos, test_pos = train_test_split(pos, test_size=0.3, random_state=42, shuffle=True)

    X_pos_strings_train = train_pos['query'].to_numpy()
    X_pos_strings_test = test_pos['query'].to_numpy()
    Y_pos_class_train = train_pos['query-label-expert'].to_numpy()
    Y_pos_class_test = test_pos['query-label-expert'].to_numpy()

    # Load other sentences
    neg = pd.concat([neg, med_neg])
    neg['query-label-expert'] = neg['query-label-expert'].fillna(1)

    train_neg, test_neg = train_test_split(neg, test_size=0.3, random_state=42, shuffle=True)

    X_neg_strings_train = train_neg['query'].to_numpy()
    X_neg_strings_test = test_neg['query'].to_numpy()
    Y_neg_class_train = train_neg['query-label-expert'].to_numpy()
    Y_neg_class_test = test_neg['query-label-expert'].to_numpy()

    #return possibly  X_concat_strings_train, X_concat_strings_test, Y_concat_class_train, Y_concat_class_test
    return X_pos_strings_train, Y_pos_class_train, X_pos_strings_test, Y_pos_class_test, X_neg_strings_train,  Y_neg_class_train, X_neg_strings_test, Y_neg_class_test

# data = pre_process()
# for i in data:
#     print(i.shape)

from sentence_transformers import SentenceTransformer

#function which embeds as expected, and then also aligns according to SVD
def embed_and_align(X_pos_strings_train, X_neg_strings_train, X_pos_strings_test, X_neg_strings_test):
    #embed all data - do I want to be able to switch this in and out?
    encoder = SentenceTransformer(f'all-MiniLM-L6-v2')
    X_pos_train_embed = encoder.encode(X_pos_strings_train, show_progress_bar=False)
    X_neg_train_embed = encoder.encode(X_neg_strings_train, show_progress_bar=False)
    X_pos_test_embed = encoder.encode(X_pos_strings_test, show_progress_bar=False)
    X_neg_test_embed = encoder.encode(X_neg_strings_test, show_progress_bar=False)
    #concat here or no?
    
    #SVD matrix found only on positive set (which is most important part of the data) of training set, but apply rotation to all data
    #is there a better way to do SVD?
    u, s, vh = np.linalg.svd(a=X_pos_train_embed)
    align_matrix = np.linalg.solve(a=vh, b=np.eye(len(X_pos_train_embed[0])))

    #perform alignments
    X_pos_train_embed_align = np.matmul(X_pos_train_embed, align_matrix)
    X_neg_train_embed_align = np.matmul(X_neg_train_embed, align_matrix)
    X_pos_test_embed_align = np.matmul(X_pos_test_embed, align_matrix)
    X_neg_test_embed_align = np.matmul(X_neg_test_embed, align_matrix)
    return X_pos_train_embed_align, X_pos_test_embed_align, X_neg_train_embed_align, X_neg_test_embed_align

# X_pos_strings_train, Y_pos_class_train, X_pos_strings_test, Y_pos_class_test, X_neg_strings_train,  Y_neg_class_train, X_neg_strings_test, Y_neg_class_test = pre_process()
# embedded_and_aligned_data = embed_and_align(X_pos_strings_train, X_neg_strings_train, X_pos_strings_test, X_neg_strings_test)
# for i in embedded_and_aligned_data:
#     print(i.shape)

def PCA_to_reduce_embeddings(X_pos_train_embed_align, X_pos_test_embed_align, X_neg_train_embed_align, X_neg_test_embed_align):
    #input size was selected as 30, but wonder how this varies!
    input_size = 30
    #we want to add up all of the data in the training set and perform PCA on all of it (for some reason ANTONIO does test set too but isn't that data leakage?)
    all_x_for_train = np.vstack([X_pos_train_embed_align, X_neg_train_embed_align])
    # PCA data
    data_pca = PCA(n_components=input_size).fit(all_x_for_train)

    #transforms each bit separately
    X_pos_train_PCA = data_pca.transform(X_pos_train_embed_align)
    X_neg_train_PCA = data_pca.transform(X_neg_train_embed_align)
    X_pos_test_PCA = data_pca.transform(X_pos_test_embed_align)
    X_neg_test_PCA = data_pca.transform(X_neg_test_embed_align)
    return X_pos_train_PCA, X_pos_test_PCA, X_neg_train_PCA, X_neg_test_PCA

# X_pos_train_embed_align, X_pos_test_embed_align, X_neg_train_embed_align, X_neg_test_embed_align = embed_and_align(X_pos_strings_train, X_neg_strings_train, X_pos_strings_test, X_neg_strings_test)
# PCA_data = PCA_to_reduce_embeddings(X_pos_train_embed_align, X_pos_test_embed_align, X_neg_train_embed_align, X_neg_test_embed_align)
# for i in PCA_data:
#     print(i.shape)

from tensorflow import keras
import tensorflow as tf

def get_model():
    #good to keep model defined separately, initialiser seed and input_size specified in here, could take out to allow lots of runs
    input_size = 30
    initializer = tf.keras.initializers.GlorotUniform(seed=42)
    inputs = keras.Input(shape=(input_size,), name="embeddings")
    x = keras.layers.Dense(128, activation="relu", kernel_initializer=initializer, name="dense_1")(inputs)
    outputs = keras.layers.Dense(2, activation="linear", kernel_initializer=initializer, name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model


def train_base_model(X_pos_train_PCA, X_pos_test_PCA, X_neg_train_PCA, X_neg_test_PCA, Y_pos_class_train, Y_pos_class_test, Y_neg_class_train, Y_neg_class_test):
    #select what you would like:
    batch_size = 64
    epochs = 30
    
    #we want to combine the X data and the Y data - use concatenate so that 1D Y data doesn't get turned into column vectors which won't like later functions
    X_train = np.concatenate((X_pos_train_PCA, X_neg_train_PCA), axis=0)
    X_test = np.concatenate((X_pos_test_PCA, X_neg_test_PCA), axis=0)
    Y_train = np.concatenate((Y_pos_class_train, Y_neg_class_train), axis=0)
    Y_test = np.concatenate((Y_pos_class_test, Y_neg_class_test), axis=0)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

    #trains on batches and uses test dataset for validation data, 
    #not great practice maybe worth trying to implement k-folds especially if doing hyperparameter tuning
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy()

    model_base = get_model()
    model_base.compile(optimizer=optimizer, loss=loss_fn, metrics=[accuracy_fn])
    model_base.fit(train_dataset, epochs=epochs, validation_data=test_dataset,)


    return model_base, X_train, X_test, Y_train, Y_test, train_dataset, test_dataset


from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def print_metrics(model, X_test, Y_test):
    # Get predicted probabilities for all classes
    y_pred_prob = model.predict(X_test)

    # Get predicted class labels (highest probability class)
    y_pred_class = np.argmax(y_pred_prob, axis=1)

    # Calculate precision, recall, and F1-score (using macro average)
    precision = precision_score(Y_test, y_pred_class, average='macro')
    recall = recall_score(Y_test, y_pred_class, average='macro')
    f1 = f1_score(Y_test, y_pred_class, average='macro')

    # Display the macro/micro/weighted average metrics
    print(f'Precision (macro): {precision:.4f}')
    print(f'Recall (macro): {recall:.4f}')
    print(f'F1-score (macro): {f1:.4f}')

    # # Binarize the output (needed for multiclass ROC)
    # # This turns the class labels into a one-vs-rest binary format
    # #here we only look at 2 classes but could add more
    # y_test_bin = label_binarize(Y_test, classes=np.arange(2))

    # # Compute ROC curve and AUC for each class
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()

    # #here we only look at 2 classes but could add more
    # for i in range(2):
    #     fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    #     roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_pred_prob[:, i])

    # # Plot the ROC curve for each class
    # plt.figure(figsize=(6, 5))
    # #here we only look at 2 classes but could add more
    # for i in range(2):
    #     plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    # plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal line
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) for Each Class')
    # plt.legend(loc='lower right')
    # plt.show()
    return 


X_pos_strings_train, Y_pos_class_train, X_pos_strings_test, Y_pos_class_test, X_neg_strings_train,  Y_neg_class_train, X_neg_strings_test, Y_neg_class_test = pre_process()
# X_pos_train_embed_align, X_pos_test_embed_align, X_neg_train_embed_align, X_neg_test_embed_align = embed_and_align(X_pos_strings_train, X_neg_strings_train, X_pos_strings_test, X_neg_strings_test)
# X_pos_train_PCA, X_pos_test_PCA, X_neg_train_PCA, X_neg_test_PCA = PCA_to_reduce_embeddings(X_pos_train_embed_align, X_pos_test_embed_align, X_neg_train_embed_align, X_neg_test_embed_align)

# get_model()
# model_base, X_base_train, X_base_test, Y_base_train, Y_base_test, train_base_dataset, test_base_dataset = train_base_model(X_pos_train_PCA, X_pos_test_PCA, X_neg_train_PCA, X_neg_test_PCA, Y_pos_class_train, Y_pos_class_test, Y_neg_class_train, Y_neg_class_test)

# print_metrics(model_base, X_base_test, Y_base_test)

import random
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer

#for LIME lets just play with X_pos_strings_train
encoder = SentenceTransformer('all-MiniLM-L6-v2')
X_pos_train_embed_align, X_pos_test_embed_align, X_neg_train_embed_align, X_neg_test_embed_align = embed_and_align(X_pos_strings_train, X_neg_strings_train, X_pos_strings_test, X_neg_strings_test)
X_pos_train_PCA, X_pos_test_PCA, X_neg_train_PCA, X_neg_test_PCA = PCA_to_reduce_embeddings(X_pos_train_embed_align, X_pos_test_embed_align, X_neg_train_embed_align, X_neg_test_embed_align)
get_model()
model_base, X_base_train, X_base_test, Y_base_train, Y_base_test, train_base_dataset, test_base_dataset = train_base_model(X_pos_train_PCA, X_pos_test_PCA, X_neg_train_PCA, X_neg_test_PCA, Y_pos_class_train, Y_pos_class_test, Y_neg_class_train, Y_neg_class_test)

u, s, vh = np.linalg.svd(encoder.encode(X_pos_strings_train, show_progress_bar=False))
align_matrix = np.linalg.solve(vh, np.eye(vh.shape[0]))
pca = PCA(n_components=30).fit(np.vstack([X_pos_train_embed_align, X_neg_train_embed_align]))

def predict_fn(texts):
    """
    texts: list of strings
    returns: np.array of shape (len(texts), num_classes) with probabilities
    """
    # Embed
    embeds = encoder.encode(texts, show_progress_bar=False)
    
    # Align embeddings (same as training)
    embeds_align = np.matmul(embeds, align_matrix)
    
    # PCA transform
    embeds_PCA = pca.transform(embeds_align)
    
    # Predict with model and convert logits to probabilities
    logits = model_base.predict(embeds_PCA)
    probs = tf.nn.softmax(logits, axis=1).numpy()
    return probs


# 1. Create a LIME explainer
explainer = LimeTextExplainer(class_names = ['medical query', 'not medical query'])  # match your classes

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

# 4. Print the explanation for the predicted class
predicted_class = np.argmax(predict_fn([example_text])[0])
print(f"Random index: {random_idx}")
print(f"Example text: {example_text}")
print(f"Predicted class: {predicted_class}")
print(f"Probability: {predict_fn([example_text])}")
print("LIME explanation:", exp.as_list(label=predicted_class))
exp_map = exp.as_map()[predicted_class]  
exp.save_to_file('data/oi.html')



