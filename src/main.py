import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from data import pre_process, embed_and_align, embed_and_align_p, PCA_to_reduce_embeddings
from sentence_transformers import SentenceTransformer





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
X_pos_train_embed_align, X_pos_test_embed_align, X_neg_train_embed_align, X_neg_test_embed_align, align_matrix= embed_and_align(X_pos_strings_train, X_neg_strings_train, X_pos_strings_test, X_neg_strings_test)
X_pos_train_PCA, X_pos_test_PCA, X_neg_train_PCA, X_neg_test_PCA, data_pca1 = PCA_to_reduce_embeddings(X_pos_train_embed_align, X_pos_test_embed_align, X_neg_train_embed_align, X_neg_test_embed_align)

get_model()
model_base, X_base_train, X_base_test, Y_base_train, Y_base_test, train_base_dataset, test_base_dataset = train_base_model(X_pos_train_PCA, X_pos_test_PCA, X_neg_train_PCA, X_neg_test_PCA, Y_pos_class_train, Y_pos_class_test, Y_neg_class_train, Y_neg_class_test)
#lets look at a few of these
# print(X_base_test.shape)
# logits = model_base.predict(X_base_test[799:800])
# prediction_probs = tf.nn.softmax(logits, axis=1).numpy()
# X_base_string_test = np.concatenate((X_pos_strings_test, X_neg_strings_test), axis=0)
# print("predictions shape:", prediction_probs.shape)
# print(X_base_string_test[799:800])
# print(prediction_probs)
#print_metrics(model_base, X_base_test, Y_base_test)

import random
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer



def LIME():

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
    exp.save_to_file('data/oi.html')


    
    return



def predict_fn(texts):

        #for LIME lets just play with X_pos_strings_train
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    X_pos_train_embed_align, X_pos_test_embed_align, X_neg_train_embed_align, X_neg_test_embed_align, align_matrix = embed_and_align(X_pos_strings_train, X_neg_strings_train, X_pos_strings_test, X_neg_strings_test)
    X_pos_train_PCA, X_pos_test_PCA, X_neg_train_PCA, X_neg_test_PCA, data_pca_l = PCA_to_reduce_embeddings(X_pos_train_embed_align, X_pos_test_embed_align, X_neg_train_embed_align, X_neg_test_embed_align)
    get_model()
    model_base, X_base_train, X_base_test, Y_base_train, Y_base_test, train_base_dataset, test_base_dataset= train_base_model(X_pos_train_PCA, X_pos_test_PCA, X_neg_train_PCA, X_neg_test_PCA, Y_pos_class_train, Y_pos_class_test, Y_neg_class_train, Y_neg_class_test)

    u, s, vh = np.linalg.svd(encoder.encode(X_pos_strings_train, show_progress_bar=False))
    align_matrix = np.linalg.solve(vh, np.eye(vh.shape[0]))
    pca = PCA(n_components=30).fit(np.vstack([X_pos_train_embed_align, X_neg_train_embed_align]))
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


#LIME()
#using only positive training data, as is done in ANTONIO for the hyper rectangles
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
        
        # Ensure the adversarial examples are within the valid input range [0, 1]
        X_adv = tf.clip_by_value(X_adv, 0.0, 1.0)
    
    return X_adv



# X_adv_test = pgd_attack_embedded(model_base, X_pos_train_PCA, Y_pos_class_train)
# print(X_pos_train_PCA.shape)
# print(X_adv_test.shape)

# # Evaluate the model on the adversarial examples
# y_pred_adv = np.argmax(model_base.predict(X_adv_test), axis=1)
# accuracy_adv = np.mean(y_pred_adv == Y_pos_class_train)

# y_pred_train = np.argmax(model_base.predict(X_pos_train_PCA), axis=1)
# accuracy_train = np.mean(y_pred_train == Y_pos_class_train)

# print(f'Accuracy on adversarial examples: {accuracy_adv:.4f}')
# print(f'Accuracy on original positive only trained examples: {accuracy_train:.4f}')

#lets now do perturbations to build the other hyper rectangles
from perturbations import create_perturbations
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
# print(X_pos_strings_train.shape)
# print(X_pos_strings_train[0])

X_train_pos_p, Y_train_pos_p, train_pos_index, X_train_neg_p, Y_train_neg_p, train_neg_index, X_test_pos_p, Y_test_pos_p, test_pos_index, X_test_neg_p, Y_test_neg_p, test_neg_index = create_perturbations(X_pos_strings_train, Y_pos_class_train, X_neg_strings_train, Y_neg_class_train, X_pos_strings_test, Y_pos_class_test, X_neg_strings_test, Y_neg_class_test, 'character')

# print(X_train_pos_p[0])
# print(Y_train_pos_p[0])
# print(train_pos_index[0])
# print(X_train_neg_p[0])
# print(Y_train_neg_p[0])
# print(train_neg_index[0])
# print(X_test_pos_p[0])
# print(Y_test_pos_p[0])
# print(test_pos_index[0])
# print(X_test_neg_p[0])
# print(Y_test_neg_p[0])
# print(test_neg_index[0])

#need to embed these perturbed versions!
Xp_pos_train_embed_align, Xp_pos_test_embed_align, Xp_neg_train_embed_align, Xp_neg_test_embed_align= embed_and_align_p(X_train_pos_p, X_train_neg_p, X_test_pos_p, X_test_neg_p, align_matrix)
#PCA but with the same fit as used for non-perturbed data
Xp_pos_train_PCA = data_pca1.transform(Xp_pos_train_embed_align)

#print(train_pos_index[:10])
#print("Expected indices:", list(range(len(X_pos_train_PCA))))



from hyperrectangles import load_hyperrectangles
hyperrectangles = load_hyperrectangles(X_pos_train_embed_align, X_pos_train_PCA, Xp_pos_train_embed_align, Xp_pos_train_PCA, train_pos_index)

print("Number of hyperrectangles:", len(hyperrectangles))
#print("Shape of first hyperrectangle:", hyperrectangles[0].shape)



