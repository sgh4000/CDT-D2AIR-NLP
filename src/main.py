import tensorflow as tf
import numpy as np
from data_preprocessing import load_data, process_and_save_data
from embedding import load_embeddings
from train import train_base
import pandas as pd
import pandas as pd
import numpy as np
import tensorflow as tf

# Model as seen here: https://github.com/Tgl70/DAIR-course-NLP/blob/main/main.py
def get_model(input_size):
    initializer = tf.keras.initializers.GlorotUniform(seed=42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_size,), name='input_features'),
        tf.keras.layers.Dense(128, activation='relu', kernel_initializer=initializer, name='dense_1'),
        tf.keras.layers.Dense(2, activation='linear', kernel_initializer=initializer, name='output_layer')
    ])
    return model

def save_test_predictions(model, X_test, y_test, raw_queries, filename, class_names):
    # Get predictions for all test queries
    logits = model.predict(X_test)
    probs = tf.nn.softmax(logits, axis=1).numpy() 

    # Predicted labels (0 = non-medical, 1 = medical)
    preds = np.argmax(probs, axis=1)

    # Build dataframe
    df = pd.DataFrame({
        "query": raw_queries,
        "true_label": y_test,
        "predicted_label": preds,
        "predicted_class": [class_names[i] for i in preds],
        "true_class": [class_names[i] for i in y_test],
        "prob_non_medical": probs[:, 0],
        "prob_medical": probs[:, 1]
    })

    df.to_csv(filename, index=False)
    
def save_dataset_to_csv(dataset, filename, class_names=None):
    all_X, all_y = [], []
    for X_batch, y_batch in dataset:
        all_X.append(X_batch.numpy())
        all_y.append(y_batch.numpy())
    
    X_arr = np.vstack(all_X)
    y_arr = np.concatenate(all_y)

    df = pd.DataFrame(X_arr)
    df["label"] = y_arr

    if class_names:
        df["label_name"] = df["label"].apply(lambda i: class_names[i])

    df.to_csv(filename, index=False)
    print(f"Saved dataset with {len(df)} rows to {filename}")

if __name__ == '__main__':
    # Set up values needing for embedding and training
    dataset_name = "medical_query_dataset"
    encoding_model = "all-MiniLM-L6-v2"
    encoding_model_name = "all-MiniLM-L6-v2"
    path = 'datasets'
    batch_size = 64
    epochs = 30
    class_names = ["non-medical", "medical"]

    # Run data preprocessing functions
    process_and_save_data()
    loaded_data = load_data()

    # Load clean and perturbed training sets

    # Clean embeddings
    X_train_o, X_test, y_train, y_test, raw_X_test = load_embeddings(
        dataset_name, encoding_model, encoding_model_name, "original",
        load_saved_embeddings=False, load_saved_align_mat=False, data=loaded_data, path=path
    )

    # Perturbed embeddings (training data perturbed, test data still clean) - 1 perturbation
    X_train_p_1, _, y_train_p_1, _ , _= load_embeddings(
        dataset_name, encoding_model, encoding_model_name, "character",n_perturbations=1,
        load_saved_embeddings=False, load_saved_align_mat=False, data=loaded_data, path=path
    )

    # Perturbed embeddings (training data perturbed, test data still clean) - 5 perturbations
    X_train_p_5, _, y_train_p_5, _, _ = load_embeddings(
        dataset_name, encoding_model, encoding_model_name, "character",n_perturbations=5,
        load_saved_embeddings=False, load_saved_align_mat=False, data=loaded_data, path=path
    )

    # Prepare datasets
    y_train = np.ravel(y_train).astype(np.int32)
    y_test = np.ravel(y_test).astype(np.int32)
    y_train_p_1 = np.ravel(y_train_p_1).astype(np.int32)
    y_train_p_5 = np.ravel(y_train_p_1).astype(np.int32)

    train_dataset_clean = tf.data.Dataset.from_tensor_slices((X_train_o, y_train)).shuffle(1024).batch(batch_size)
    train_dataset_pert_1 = tf.data.Dataset.from_tensor_slices((X_train_p_1, y_train_p_1)).shuffle(1024).batch(batch_size)
    train_dataset_pert_5 = tf.data.Dataset.from_tensor_slices((X_train_p_5, y_train_p_5)).shuffle(1024).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    save_dataset_to_csv(test_dataset, "test_dataset.csv", class_names=["medical", "non-medical"])

    input_size = X_train_o.shape[1]

    # Train model on clean data
    model_clean = get_model(input_size)
    model_clean.save("models/medical_query_clean.keras")

    model_clean = train_base(model_clean, train_dataset_clean, test_dataset, epochs, seed=42)
    loss_c, acc_c = model_clean.evaluate(X_test, y_test, verbose=0)
   
    # Train model on perturbed data - 1 perturbation
    model_pert_1 = get_model(input_size)
    model_pert_1 = train_base(model_pert_1, train_dataset_pert_1, test_dataset, epochs, seed=42)
    loss_p_1, acc_p_1 = model_pert_1.evaluate(X_test, y_test, verbose=0)
    
    # Train model on perturbed data - 5 perturbations
    model_pert_5 = get_model(input_size)
    model_pert_5 = train_base(model_pert_5, train_dataset_pert_5, test_dataset, epochs, seed=42)
    loss_p_5, acc_p_5 = model_pert_5.evaluate(X_test, y_test, verbose=0)

    # Save models so don't need to keep retraining
    model_clean.save("models/medical_query_clean")
    model_pert_1.save("models/medical_query_pert_1")
    model_pert_5.save("models/medical_query_pert_5")    

    print(f"\nClean model → Test accuracy: {acc_c:.3f}")
    print(f"Perturbed model - 1 pert → Test accuracy: {acc_p_1:.3f}")
    print(f"Perturbed model - 5 pert → Test accuracy: {acc_p_5:.3f}")

    # Saving test prediction data for future inspection

    save_test_predictions(model_clean, X_test, y_test, raw_X_test, "test_predictions_clean.csv", class_names)
    save_test_predictions(model_pert_1, X_test, y_test, raw_X_test, "test_predictions_pert_1.csv", class_names)
    save_test_predictions(model_pert_5, X_test, y_test, raw_X_test, "test_predictions_pert_5.csv", class_names)