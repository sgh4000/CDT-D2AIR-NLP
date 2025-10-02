import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

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
    pos = expert.loc[(expert['query-label-expert'] >= 1) & (expert['query-label-expert'] <= 3)].copy()
    pos['query-label-expert'] = pos['query-label-expert'].replace({1: 0, 2: 0, 3: 0})

    neg = expert.loc[expert['query-label-expert'] == 0].copy()
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
    return X_pos_train_embed_align, X_pos_test_embed_align, X_neg_train_embed_align, X_neg_test_embed_align, align_matrix

#function which embeds as expected, and then also aligns according to SVD
def embed_and_align_p(X_pos_strings_train, X_neg_strings_train, X_pos_strings_test, X_neg_strings_test, align_matrix):
    #embed all data - do I want to be able to switch this in and out?
    encoder = SentenceTransformer(f'all-MiniLM-L6-v2')
    X_pos_train_embed = encoder.encode(X_pos_strings_train, show_progress_bar=False)
    X_neg_train_embed = encoder.encode(X_neg_strings_train, show_progress_bar=False)
    X_pos_test_embed = encoder.encode(X_pos_strings_test, show_progress_bar=False)
    X_neg_test_embed = encoder.encode(X_neg_strings_test, show_progress_bar=False)
    #concat here or no?
    

    #perform alignments based on precomputed matrix
    X_pos_train_embed_align = np.matmul(X_pos_train_embed, align_matrix)
    X_neg_train_embed_align = np.matmul(X_neg_train_embed, align_matrix)
    X_pos_test_embed_align = np.matmul(X_pos_test_embed, align_matrix)
    X_neg_test_embed_align = np.matmul(X_neg_test_embed, align_matrix)
    return X_pos_train_embed_align, X_pos_test_embed_align, X_neg_train_embed_align, X_neg_test_embed_align

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
    return X_pos_train_PCA, X_pos_test_PCA, X_neg_train_PCA, X_neg_test_PCA, data_pca

    