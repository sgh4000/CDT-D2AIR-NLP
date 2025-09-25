import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# This is a script to work out TN, FP, FN, TP

def analyse_results(csv_path, n_examples=5):
    
    df = pd.read_csv(csv_path)

    y_true = df["true_label"].astype(int)
    y_pred = df["predicted_label"].astype(int)

    # Applying the scikitlearn confusion matrix https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print("-- Confusion Matrix (labels: 0=non-medical, 1=medical) --")
    print(f"TN (correct non-medical): {tn}")
    print(f"FP (non-medical → medical): {fp}")
    print(f"FN (medical → non-medical): {fn}")
    print(f"TP (correct medical): {tp}")
    print()

    # Metrics
    print("-- Classification Report --")
    print(classification_report(y_true, y_pred, target_names=["non-medical", "medical"]))
    print()

    # Per-class accuracy
    non_med_mask = y_true == 0
    med_mask = y_true == 1
    acc_non_med = accuracy_score(y_true[non_med_mask], y_pred[non_med_mask])
    acc_med = accuracy_score(y_true[med_mask], y_pred[med_mask])
    print("-- Per-Class Accuracy --")
    print(f"Non-medical accuracy: {acc_non_med:.3f}")
    print(f"Medical accuracy:     {acc_med:.3f}")
    print(f"Overall accuracy:     {accuracy_score(y_true, y_pred):.3f}")
    print()

    # Example queries
    print(f"-- Example False Positives (non-medical → medical) --")
    fp_examples = df[(df["true_label"] == 0) & (df["predicted_label"] == 1)].head(n_examples)
    for _, row in fp_examples.iterrows():
        print(f"Query: {row['query']} | True: non-medical | Pred: medical | "
              f"Probs: med={row['prob_medical']:.3f}, non-med={row['prob_non_medical']:.3f}")

    print(f"\n-- Example False Negatives (medical → non-medical) --")
    fn_examples = df[(df["true_label"] == 1) & (df["predicted_label"] == 0)].head(n_examples)
    for _, row in fn_examples.iterrows():
        print(f"Query: {row['query']} | True: medical | Pred: non-medical | "
              f"Probs: med={row['prob_medical']:.3f}, non-med={row['prob_non_medical']:.3f}")

    print(f"\n-- Example True Negatives (correct non-medical) --")
    tn_examples = df[(df["true_label"] == 0) & (df["predicted_label"] == 0)].head(n_examples)
    for _, row in tn_examples.iterrows():
        print(f"Query: {row['query']} | True: non-medical | Pred: non-medical | "
              f"Probs: med={row['prob_medical']:.3f}, non-med={row['prob_non_medical']:.3f}")

if __name__ == "__main__":
    analyse_results("test_predictions_clean.csv")
