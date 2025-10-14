import numpy as np
from data import pre_process, embed_and_align, embed_and_align_p, PCA_to_reduce_embeddings
from train import get_model, train_base_model, adv_epsilon_training, adv_hyperrectangles_training
from metrics import print_metrics, confusion_matrix_display, generalisability_metric, robustness_metric
from sentence_transformers import SentenceTransformer
from explainability import lime_test, make_predict_fn, SHAP_vis
from pgd_attack import pgd_attack_epsilon, pgd_attack_hyperrectangles
from perturbations import create_perturbations
from hyperrectangles import load_hyperrectangles
from visualise import tsne_display, UMAP_display, data_balance_display, print_cosine_sim_demo, print_silhouette_score, UMAP_investigate


#############################################
#### Pre-processing and Visualising Data ####
#############################################

#Preprocess data: Train vs Test, and Pos (Pos medical query = 0) vs Neg (Neg medical query = 1) (this is needed to compute perturbations for positive class only later)
X_pos_strings_train, Y_pos_class_train, X_pos_strings_test, Y_pos_class_test, X_neg_strings_train, Y_neg_class_train, X_neg_strings_test, Y_neg_class_test = pre_process()

#View the balance of data, there appears to be nearly equal of each class which is beneficial for training
data_balance_display(X_pos_strings_train, X_neg_strings_train, X_pos_strings_test, X_neg_strings_test)

# PLEASE BE AWARE THIS IS WHERE I SWITCH FROM TRAIN (P N) TEST (P N) TO POS (TN TS) NEG (TN TS) WATCH OUT
#Like in ANTONIO, use SVD to align the entire dataset in the orientation based on positive medical query training data
#used same encoder as ANTONIO, but important to edit inside this function AND within this script so that LIME is correct, if changed
X_pos_train_embed_align, X_pos_test_embed_align, X_neg_train_embed_align, X_neg_test_embed_align, align_matrix= embed_and_align(X_pos_strings_train, X_neg_strings_train, X_pos_strings_test, X_neg_strings_test)

#Concatenate datasets for visualisation tasks - could do in another location but want flexibility
Vis_X_train_string = np.concatenate((X_pos_strings_train, X_neg_strings_train), axis=0)
Vis_X_train_embed_align = np.concatenate((X_pos_train_embed_align, X_neg_train_embed_align), axis=0)
Vis_X_test_string = np.concatenate((X_pos_strings_test, X_neg_strings_test), axis=0)
Vis_X_test_embed_align = np.concatenate((X_pos_test_embed_align, X_neg_test_embed_align), axis=0)
Vis_Y_train = np.concatenate((Y_pos_class_train, Y_neg_class_train), axis=0)
Vis_Y_test = np.concatenate((Y_pos_class_test, Y_neg_class_test), axis=0)

#Silhouette score gives an indication of how separate the clusters - output between -1 to 1 and we want it to be as high as possible
print_silhouette_score(Vis_X_train_embed_align)
#this value near 0 indicates there are samples on the border of clusters
#this next step I made to look at some examples of strings with a high cosine similarity as a sanity check
print_cosine_sim_demo(Vis_X_train_embed_align, Vis_X_train_string, Vis_Y_train)
#note that results from this indicate how embeddings don't seem to hold semantic meaning very well here - similarity is computed low for similar sentneces, or high with misclassing

#Two popular methods of visually inspecting high-D data are t-SNE and UMAP, for completeness I included both, but for speed and to see how distance might relate to meaning will only show UMAP
#Be aware that UMAP assumes embedding data lies on a manifold and this may not be accurate
# tsne_title1 = "T-SNE of training data after embeddings and SVD"
# tsne_display(Vis_X_train_embed_align, Vis_Y_train, tsne_title1)
umap_title1 = "UMAP of training data after embeddings and SVD"
UMAP_output = UMAP_display(Vis_X_train_embed_align, Vis_Y_train, umap_title1)

UMAP_investigate(UMAP_output, Vis_X_train_string, Vis_Y_train)
#from the investigation it is clear there is a little bit of ambiguity in whether something is a medical query, bodily query, formed like a question etc

#Based on ANTONIO code, a PCA was then performed to reduce the dimensionality of the data
#this is common in embeddings, but it would be useful to visualise the variance in a graph, or tune this to see what produces the best accuracy
X_pos_train_PCA, X_pos_test_PCA, X_neg_train_PCA, X_neg_test_PCA, data_pca1 = PCA_to_reduce_embeddings(X_pos_train_embed_align, X_pos_test_embed_align, X_neg_train_embed_align, X_neg_test_embed_align)

# #could visualise how this affects the UMAP and t-SNE plots - commented out here for brevity
# Vis_X_train_PCA = np.concatenate((X_pos_train_PCA, X_neg_train_PCA), axis=0)
# Vis_X_test_PCA = np.concatenate((X_pos_test_PCA, X_neg_test_PCA), axis=0)
# # tsne_title2 = "T-SNE of training data after PCA"
# # tsne_display(Vis_X_train_PCA, Vis_Y_train, tsne_title2)
# umap_title2 = "UMAP of training data after PCA"
# UMAP_display(Vis_X_train_PCA, Vis_Y_train, umap_title2)


##################
#### Training ####
##################

#we pick the model we would like, and train a base model! Based off of ANTONIO and Katya's Lab
get_model()
model_base, X_base_train, X_base_test, Y_base_train, Y_base_test, train_base_dataset, test_base_dataset = train_base_model(X_pos_train_PCA, X_pos_test_PCA, X_neg_train_PCA, X_neg_test_PCA, Y_pos_class_train, Y_pos_class_test, Y_neg_class_train, Y_neg_class_test)

#################
#### Metrics ####
#################


#lets compute metrics for the best tests, generate a confusion matrix, and see how well the model generalises via assessing balanced accuracy 
#prints accuracy, balanced accuracy, recall, precision, F1 and an ROC curve
print_metrics(model_base, X_base_test, Y_base_test)
#generates a confusion matrix and displays it, outputs confusion matrix for later assessment of TP,TN,FP,FN
cm = confusion_matrix_display(model_base, X_base_test, Y_base_test)
#simple metric with 0.05 comparison chosen randomly - how could I pick this better?
generalisability_metric(model_base, X_base_train, Y_base_train, X_base_test, Y_base_test)
#need encoder here for LIME but could also but this into the beginning and for the embeddings so different ones can be tried!

########################
#### Explainability ####
########################

#in order to use LIME, have to create a predict_fn function, however this cannot have any other inputs than the original text string
#hence this is created by a second predict_fn function which takes all the important matrices for SVD, PCA, the model and the chosen encoder
#if anything is changed in model pipeline please review explainability script!

predict_fn= make_predict_fn(align_matrix, data_pca1, model_base)
random_idx = lime_test(Vis_X_test_string, predict_fn)

#trying to use SHAP and using the same random index as LIME
SHAP_vis(predict_fn, Vis_X_test_string, random_idx)


# ############################
# #### Adversarial Attack ####
# ############################

# ######################
# #### Epsilon Ball ####
# ######################

# #Simple epsilon ball pgd attack based on Katya's code, but only on positive data like in ANTONIO
# X_adv = pgd_attack_epsilon(model_base, X_pos_train_PCA, Y_pos_class_train)

# # Evaluate the model on the adversarial examples
# # Could I use print_metrics function here?
# #uses logits, but should perhaps convert to softmax probabilities
# y_pred_adv = np.argmax(model_base.predict(X_adv), axis=1)
# accuracy_adv = np.mean(y_pred_adv == Y_pos_class_train)

# #uses logits, but should perhaps convert to softmax probabilities
# y_pred_train = np.argmax(model_base.predict(X_pos_train_PCA), axis=1)
# accuracy_train = np.mean(y_pred_train == Y_pos_class_train)

# print(f'Accuracy on adversarial examples: {accuracy_adv:.4f}')
# print(f'Accuracy on original positive only trained examples: {accuracy_train:.4f}')

# #created a robustness metric, same as generalisability but instead of for test data, uses adversarial data generated for positive X training data
# robustness_metric(model_base, X_base_train, Y_base_train, X_adv, Y_pos_class_train)

# ##########################
# #### Hyper-rectangles ####
# ##########################

# #This is lifted from ANTONIO, but inside function all other types of perturbation training was removed - 'word' because of conflicts with requirements, vicuna for space
# X_train_pos_p, Y_train_pos_p, train_pos_index, X_train_neg_p, Y_train_neg_p, train_neg_index, X_test_pos_p, Y_test_pos_p, test_pos_index, X_test_neg_p, Y_test_neg_p, test_neg_index = create_perturbations(X_pos_strings_train, Y_pos_class_train, X_neg_strings_train, Y_neg_class_train, X_pos_strings_test, Y_pos_class_test, X_neg_strings_test, Y_neg_class_test, 'character')

# #Embed and Align perturbed version but using SVD matrix from original base training data to compare
# Xp_pos_train_embed_align, Xp_pos_test_embed_align, Xp_neg_train_embed_align, Xp_neg_test_embed_align= embed_and_align_p(X_train_pos_p, X_train_neg_p, X_test_pos_p, X_test_neg_p, align_matrix)
# #PCA from original base training data to compare
# Xp_pos_train_PCA = data_pca1.transform(Xp_pos_train_embed_align)
# #generates hyperrectangles - lifted from ANTONIO code
# hyperrectangles = load_hyperrectangles(X_pos_train_embed_align, X_pos_train_PCA, Xp_pos_train_embed_align, Xp_pos_train_PCA, train_pos_index)

# #sanity check
# print("Number of hyperrectangles:", len(hyperrectangles))

# n_samples = int(len(X_pos_train_PCA))
# #generating just attack data using the hyperrectangles - lifed from inside adversarial attack in ANTONIO code
# pgd_dataset_hyper, pgd_labels_inside_hyper = pgd_attack_hyperrectangles(model_base, hyperrectangles, n_samples)
# #uses logits, but should perhaps convert to softmax probabilities
# y_hyper_pred = np.argmax(model_base.predict(pgd_dataset_hyper), axis=1)
# accuracy_hyper_pred = np.mean(y_hyper_pred == pgd_labels_inside_hyper)


# # Evaluate the model on the adversarial examples
# # Could I use print_metrics function here?
# print(f'Accuracy on adversarial attack from hyper rectangle examples: {accuracy_hyper_pred:.4f}')
# #created a robustness metric, same as generalisability but instead of for test data, uses adversarial data generated for positive X training data
# robustness_metric(model_base, X_base_train, Y_base_train, pgd_dataset_hyper, pgd_labels_inside_hyper)

# ##############################
# #### Adversarial Training ####
# ##############################

# ######################
# #### Epsilon Ball ####
# ######################

# #lifted from Katya's lab
# model_adv_eps = get_model()
# #this is dropping accuracy for train but the accuracy for test is high
# model_adv_eps = adv_epsilon_training(model_adv_eps, train_base_dataset, test_base_dataset)
# #lets evaluate on clean train data
# y_adv_pred_pos_train = np.argmax(model_adv_eps.predict(X_pos_train_PCA), axis=1)
# accuracy_train_adv_pred_pos = np.mean(y_adv_pred_pos_train == Y_pos_class_train)

# y_adv_pred_train_eps = np.argmax(model_adv_eps.predict(X_base_train), axis=1)
# accuracy_train_adv_pred_eps = np.mean(y_adv_pred_train_eps == Y_base_train)

# print(f'Accuracy on original positive only trained examples: {accuracy_train_adv_pred_pos:.4f}')
# print(f'Accuracy on original total trained examples: {accuracy_train_adv_pred_eps:.4f}')

# ##########################
# #### Hyper-rectangles ####
# ##########################

# #lifted from ANTONIO
# model_adv_hyper = get_model()
# model_adv_hyper = adv_hyperrectangles_training(model_adv_hyper, train_base_dataset, test_base_dataset, hyperrectangles, n_samples)
# #lets evaluate on clean train data
# y_adv_hyper_pred_pos_train = np.argmax(model_adv_hyper.predict(X_pos_train_PCA), axis=1)
# accuracy_train_adv_hyper_pred_pos = np.mean(y_adv_hyper_pred_pos_train == Y_pos_class_train)

# y_adv_hyper_pred_train = np.argmax(model_adv_hyper.predict(X_base_train), axis=1)
# accuracy_train_adv_hyper_pred = np.mean(y_adv_hyper_pred_train == Y_base_train)

# print(f'Hyper Accuracy on original positive only trained examples: {accuracy_train_adv_hyper_pred_pos:.4f}')
# print(f'Hyper Accuracy on original total trained examples: {accuracy_train_adv_hyper_pred:.4f}')
