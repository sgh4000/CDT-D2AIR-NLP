import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from data import pre_process, embed_and_align, embed_and_align_p, PCA_to_reduce_embeddings
from train import get_model, train_base_model
from sentence_transformers import SentenceTransformer
from tensorflow import keras
import tensorflow as tf
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