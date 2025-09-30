#here I will place the imports

#function to preprocess the raw data before embeddings
def pre_process(raw_data_expert, raw_data_neg):
    #should read the data from the files, and begin extract the relevant strings and label them
    #do I need to include severity levels, and indexing here?
    #save them where needed?
    #do i need to do train testing split here?
    return X_pos_strings_train, X_pos_strings_test, X_neg_strings_train, X_neg_strings_test, Y_pos_class_train, Y_pos_class_test, Y_neg_class_train, Y_neg_class_test, X_concat_strings_train, X_concat_strings_test, Y_concat_class_train, Y_concat_class_test

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
    return X_pos_train_embed_align, X_neg_train_embed_align, X_pos_test_embed_align, X_neg_test_embed_align

def PCA(X_pos_train_embed_align, X_neg_train_embed_align, X_pos_test_embed_align, X_neg_test_embed_align):
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
    return X_pos_train_PCA, X_neg_train_PCA, X_pos_test_PCA, X_neg_test_PCA

def train_base_model(X_pos_train_PCA, X_neg_train_PCA, X_pos_test_PCA, X_neg_test_PCA, Y_pos_class_train, Y_pos_class_test, Y_neg_class_train, Y_neg_class_test):
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


    return model

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

def print_metrics(model, x_test, y_test):
    # Get predicted probabilities for all classes
    y_pred_prob = model.predict(x_test)

    # Get predicted class labels (highest probability class)
    y_pred_class = np.argmax(y_pred_prob, axis=1)

    # Calculate precision, recall, and F1-score (using macro average)
    precision = precision_score(y_test, y_pred_class, average='macro')
    recall = recall_score(y_test, y_pred_class, average='macro')
    f1 = f1_score(y_test, y_pred_class, average='macro')

    # Display the macro/micro/weighted average metrics
    print(f'Precision (macro): {precision:.4f}')
    print(f'Recall (macro): {recall:.4f}')
    print(f'F1-score (macro): {f1:.4f}')

    # Binarize the output (needed for multiclass ROC)
    # This turns the class labels into a one-vs-rest binary format
    #here we only look at 2 classes but could add more
    y_test_bin = label_binarize(y_test, classes=np.arange(2))

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    #here we only look at 2 classes but could add more
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_pred_prob[:, i])

    # Plot the ROC curve for each class
    plt.figure(figsize=(6, 5))
    #here we only look at 2 classes but could add more
    for i in range(2):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) for Each Class')
    plt.legend(loc='lower right')
    plt.show()