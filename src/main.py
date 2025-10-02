import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from data import pre_process, embed_and_align, embed_and_align_p, PCA_to_reduce_embeddings
from train import get_model, train_base_model, adversarial_training, adv_hyperrectangles_training
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
from explainability import lime_test, make_predict_fn
from pgd_attack import pgd_attack_embedded
from perturbations import create_perturbations
from hyperrectangles import load_hyperrectangles
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
import time

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

print_metrics(model_base, X_base_test, Y_base_test)
#need encoder here for LIME but could also but this into the beginning and for the embeddings so different ones can be tried!
encoder = SentenceTransformer('all-MiniLM-L6-v2')

predict_fn = make_predict_fn(encoder, align_matrix, data_pca1, model_base)

lime_test(X_pos_strings_test, predict_fn)
#using only positive training data, as is done in ANTONIO for the hyper rectangles


#do simple epsilon ball attack here
X_adv_test = pgd_attack_embedded(model_base, X_pos_train_PCA, Y_pos_class_train)
print(X_pos_train_PCA.shape)
print(X_adv_test.shape)

# Evaluate the model on the adversarial examples
y_pred_adv = np.argmax(model_base.predict(X_adv_test), axis=1)
accuracy_adv = np.mean(y_pred_adv == Y_pos_class_train)

y_pred_train = np.argmax(model_base.predict(X_pos_train_PCA), axis=1)
accuracy_train = np.mean(y_pred_train == Y_pos_class_train)

print(f'Accuracy on adversarial examples: {accuracy_adv:.4f}')
print(f'Accuracy on original positive only trained examples: {accuracy_train:.4f}')

#lets now do perturbations to build the other hyper rectangles

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

hyperrectangles = load_hyperrectangles(X_pos_train_embed_align, X_pos_train_PCA, Xp_pos_train_embed_align, Xp_pos_train_PCA, train_pos_index)

print("Number of hyperrectangles:", len(hyperrectangles))
#print("Shape of first hyperrectangle:", hyperrectangles[0].shape)

model_adv = get_model()
#this is dropping accuracy for train but the accuracy for test is high
model_adv = adversarial_training(model_adv, train_base_dataset, test_base_dataset)
#lets evaluate on clean train data
y_adv_pred_pos_train = np.argmax(model_adv.predict(X_pos_train_PCA), axis=1)
accuracy_train_adv_pred_pos = np.mean(y_adv_pred_pos_train == Y_pos_class_train)

y_adv_pred_train = np.argmax(model_adv.predict(X_base_train), axis=1)
accuracy_train_adv_pred = np.mean(y_adv_pred_train == Y_base_train)

print(f'Accuracy on original positive only trained examples: {accuracy_train_adv_pred_pos:.4f}')
print(f'Accuracy on original total trained examples: {accuracy_train_adv_pred:.4f}')

n_samples = int(len(X_pos_train_PCA))
model_adv_hyper = get_model()
model_adv_hyper = adv_hyperrectangles_training(model_adv_hyper, train_base_dataset, test_base_dataset, hyperrectangles, n_samples)
#lets evaluate on clean train data
y_adv_hyper_pred_pos_train = np.argmax(model_adv_hyper.predict(X_pos_train_PCA), axis=1)
accuracy_train_adv_hyper_pred_pos = np.mean(y_adv_hyper_pred_pos_train == Y_pos_class_train)

y_adv_hyper_pred_train = np.argmax(model_adv_hyper.predict(X_base_train), axis=1)
accuracy_train_adv_hyper_pred = np.mean(y_adv_hyper_pred_train == Y_base_train)


print(f'Hyper Accuracy on original positive only trained examples: {accuracy_train_adv_hyper_pred_pos:.4f}')
print(f'Hyper Accuracy on original total trained examples: {accuracy_train_adv_hyper_pred:.4f}')