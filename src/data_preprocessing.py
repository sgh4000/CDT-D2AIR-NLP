import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    # Simply load in the data from the medicheck-preprocessed.csv

    data = pd.read_csv('../data/medicheck-preprocessed.csv')

    X = data["query"].tolist()
    y = data["label"].values

    # Train/Test split, 80/20 seems sensible in both medical and non-medical classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return [X_train, X_test, y_train, y_test]

def process_and_save_data():

    # This is just preprocessing the medical question datasets, to create a simple csv file in the format
    # [query,label]
    # with a label value of 1 indicating medical question, and label value of 0 indicating non-medical question

    # To run this script, please ensure you have downloaded the datasets to a /data file in the top level of the repo directory:
    # medicheck-expert.csv https://github.com/GavinAbercrombie/medical-safety/blob/main/data/medicheck-expert.csv
    # medicheck-neg.csv https://github.com/GavinAbercrombie/medical-safety/blob/main/data/medicheck-neg.csv


    # Load expert sourced medical dataset
    df_expert = pd.read_csv("../data/medicheck-expert.csv", usecols=["query"])
    df_expert["query"] = df_expert["query"].astype(str).str.strip().str.strip('"').str.strip("'")
    df_expert["label"] = 1

    # Load negative (non medical) dataset (just a list of questions so can read in as raw text)
    with open("../data/medicheck-neg.csv", "r", encoding="utf-8") as f:
        neg_queries = [line.strip().strip('"').strip("'") for line in f if line.strip()]

    df_neg = pd.DataFrame({"query": neg_queries})
    df_neg["label"] = 0

    # Combine to simple dataset
    df_all = pd.concat([df_expert, df_neg], ignore_index=True)

    # Prepare X,y for future embedding tasks
    X = df_all["query"].tolist()
    y = df_all["label"].values

    # Print some info on the data sets
    print("Dataset size:", len(X))
    print("Class balance:", pd.Series(y).value_counts().to_dict())
    print(df_all.head())

    # Save to new csv file for future use
    df_all.to_csv("../data/medicheck-preprocessed.csv", index=False)