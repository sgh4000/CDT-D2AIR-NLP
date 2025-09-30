#here I will place the imports
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import csv
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
    return X_pos_strings_train, X_pos_strings_test, X_neg_strings_train, X_neg_strings_test, Y_pos_class_train, Y_pos_class_test, Y_neg_class_train, Y_neg_class_test

data = pre_process()
for i in data:
    print(i[:5])


