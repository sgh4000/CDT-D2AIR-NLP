import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from data import pre_process, embed_and_align, embed_and_align_p, PCA_to_reduce_embeddings
from train import get_model, train_base_model
from sentence_transformers import SentenceTransformer
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
import random
#set seeds for reprodicibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def print_metrics(model, X_test, Y_test):
    # Get predicted probabilities for all classes
    y_pred_logits = model.predict(X_test)
    exp_logits = np.exp(y_pred_logits)
    y_pred_prob_softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


    # Get predicted class labels (highest probability class)
    y_pred_class = np.argmax(y_pred_prob_softmax, axis=1)

    # Calculate precision, recall, and F1-score (using macro average)
    accuracy = accuracy_score(Y_test, y_pred_class)
    balanced_accuracy = balanced_accuracy_score(Y_test, y_pred_class)
    precision = precision_score(Y_test, y_pred_class, average='binary', pos_label=0)
    recall = recall_score(Y_test, y_pred_class, average='binary', pos_label=0)
    f1 = f1_score(Y_test, y_pred_class, average='binary', pos_label=0)

    # Display the macro/micro/weighted average metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Balanced Accuracy: {balanced_accuracy:.4f}')
    print(f'Precision (binary): {precision:.4f}')
    print(f'Recall (binary): {recall:.4f}')
    print(f'F1-score (binary): {f1:.4f}')


    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    print(y_pred_prob_softmax)
    y_score = y_pred_prob_softmax[:,0]

    fpr, tpr, thresholds = roc_curve(Y_test, y_score, pos_label=0)
    roc_auc = 1 - roc_auc_score(Y_test, y_score) #need 1 - as automatically it assumed region of interest is 1, not 0
    print(f"ROC score for positive medical query class: {roc_auc}")

    # Plot the ROC curve for each class
    plt.figure(figsize=(10, 6))

    plt.plot(fpr, tpr, label=f'Class 0 - Medical Query (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title = 'Receiver Operating Characteristic (ROC) for Positive Medical Query Class'
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(f"data/metrics/{title}.png", dpi=300, bbox_inches='tight')
    plt.show()
    return 

def confusion_matrix_display(model, x, y):
    y_pred = model.predict(x)
    exp_logits = np.exp(y_pred)
    y_pred_prob_softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    y_pred_class = np.argmax(y_pred_prob_softmax, axis=1)
    cm = confusion_matrix(y, y_pred_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot(cmap=plt.cm.Blues)
    title = 'Confusion Matrix for Base Model'
    plt.title(title)
    plt.savefig(f"data/metrics/{title}.png", dpi=300, bbox_inches='tight')
    plt.show()
    return cm


def generalisability_metric(model, xtn, ytn, xts, yts):
    y_pred_tn = model.predict(xtn)
    exp_logits_tn = np.exp(y_pred_tn)
    y_pred_prob_softmax_tn = exp_logits_tn / np.sum(exp_logits_tn, axis=1, keepdims=True)
    y_pred_class_tn = np.argmax(y_pred_prob_softmax_tn, axis=1)
    balanced_accuracy_tn = balanced_accuracy_score(ytn, y_pred_class_tn)
    y_pred_ts = model.predict(xts)
    exp_logits_ts = np.exp(y_pred_ts)
    y_pred_prob_softmax_ts = exp_logits_ts / np.sum(exp_logits_ts, axis=1, keepdims=True)
    y_pred_class_ts = np.argmax(y_pred_prob_softmax_ts, axis=1)
    balanced_accuracy_ts = balanced_accuracy_score(yts, y_pred_class_ts)
    print(f'Balanced Accuracy for Train Data: {balanced_accuracy_tn:.4f}')
    print(f'Balanced Accuracy for Test Data: {balanced_accuracy_ts:.4f}')
    if abs(balanced_accuracy_tn-balanced_accuracy_ts)<0.05:
        print("Good generalisability as Balanced Accuracy for Train and Test Data between 0.05 difference ")
    else:
        print("Not good generalisability")
    return

def robustness_metric(model, xtn, ytn, xadv, yadv):
    y_pred_tn = model.predict(xtn)
    exp_logits_tn = np.exp(y_pred_tn)
    y_pred_prob_softmax_tn = exp_logits_tn / np.sum(exp_logits_tn, axis=1, keepdims=True)
    y_pred_class_tn = np.argmax(y_pred_prob_softmax_tn, axis=1)
    balanced_accuracy_tn = balanced_accuracy_score(ytn, y_pred_class_tn)
    y_pred_adv = model.predict(xadv)
    exp_logits_adv = np.exp(y_pred_adv)
    y_pred_prob_softmax_adv = exp_logits_adv / np.sum(exp_logits_adv, axis=1, keepdims=True)
    y_pred_class_adv = np.argmax(y_pred_prob_softmax_adv, axis=1)
    balanced_accuracy_adv = balanced_accuracy_score(yadv, y_pred_class_adv)
    print(f'Balanced Accuracy for Train Data: {balanced_accuracy_tn:.4f}')
    print(f'Balanced Accuracy for Adversarial Attack Data: {balanced_accuracy_adv:.4f}')
    if abs(balanced_accuracy_tn-balanced_accuracy_adv)<0.05:
        print("Good Robustness as Balanced Accuracy for Train and Test Data between 0.05 difference ")
    else:
        print("Not good robustness")
    return

    