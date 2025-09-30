#here I will place the imports
import numpy as np

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

def visualise(X_pos_train_embed_align, X_neg_train_embed_align, X_pos_test_embed_align, X_neg_test_embed_align, X_pos_strings_train, X_pos_strings_test, X_neg_strings_train, X_neg_strings_test, Y_pos_class_train, Y_pos_class_test, Y_neg_class_train, Y_neg_class_test, X_concat_strings_train, X_concat_strings_test, Y_concat_class_train, Y_concat_class_test, X_pos_train_PCA, X_neg_train_PCA, X_pos_test_PCA, X_neg_test_PCA):
    #bins to look at how much data is in each and to look at balance
    #t-SNE
    #UMAP nearest neighbours
    #cosine
    #silhouette
    return 

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


    return model, X_train, X_test, Y_train, Y_test, train_dataset, test_dataset

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

def print_metrics(model, X_train, X_test, Y_train, Y_test, train_dataset, test_dataset):
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

    # Binarize the output (needed for multiclass ROC)
    # This turns the class labels into a one-vs-rest binary format
    #here we only look at 2 classes but could add more
    y_test_bin = label_binarize(Y_test, classes=np.arange(2))

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
    return 

#PGD attack on training dataset via making hyperrectangles

def calculate_hyperrectangle(points):
    # Calculate the hyperrectangle around the points
    min_max_list = np.full((points.shape[1], 2), [np.inf, -np.inf])
    for point in points:
        for i in range(len(point)):
            if point[i] < min_max_list[i][0]:
                min_max_list[i][0] = point[i]
            if point[i] > min_max_list[i][1]:
                min_max_list[i][1] = point[i]
    return min_max_list

#these essentially make the spaces in which a pgd attack would occur in - eps_cube is purely geometric, and not helpful if mapping back via NLP
#hyperrectangles are better as they are based on perturbations
def create_hyperrectangles(X_pos_train_embed_align, Xp_pos_train_embed_align, X_pos_train_PCA, Xp_pos_train_PCA, train_pos_index, X_neg_train_embed_align, X_pos_test_embed_align, X_neg_test_embed_align, method, eps=0.05):
    hyperrectangles = []

    if method == 'eps_cube':
        for p in X_pos_train_PCA:
            eps_cube = []
            for d in p:
                eps_cube.append(np.array([d - eps, d + eps]))
            hyperrectangles.append(np.array(eps_cube))

    else:


        for i in range(len(X_pos_train_PCA)):
            points = []
            points.append(X_pos_train_PCA[i])
            for index, value in enumerate(train_pos_index):
                if value == i:
                    cosine_score = util.cos_sim(X_pos_train_embed_align[i], Xp_pos_train_embed_align[index])
                    if cosine_score > 0.6:
                        points.append(Xp_pos_train_PCA[index])
            if len(points) >= 2:
                hyperrectangle = calculate_hyperrectangle(np.array(points))
                hyperrectangles.append(hyperrectangle)

    print(np.array(hyperrectangles).shape)

    return hyperrectangles

#create adversarial data!

def generate_pgd_from_hyperrectangles():
    if seed:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    optimizer = keras.optimizers.Adam()
    ce_batch_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
    pgd_batch_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
    pgd_attack_single_image_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)

    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    test_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    train_loss_metric = keras.metrics.SparseCategoricalCrossentropy(from_logits=from_logits)
    test_loss_metric = keras.metrics.SparseCategoricalCrossentropy(from_logits=from_logits)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for x_batch_train, y_batch_train in train_dataset:
            # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                outputs = model(x_batch_train, training=True)  # Outputs for this minibatch
                ce_loss_value = ce_batch_loss(y_batch_train, outputs)
                ce_loss_value = ce_loss_value * alfa
            # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(ce_loss_value, model.trainable_weights)
            # Run one step of gradient descent by updating the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        #########################################PGD####################################################
        pgd_dataset = []
        np.random.shuffle(hyperrectangles)
        for hyperrectangle in hyperrectangles[:n_samples]:
            t_hyperrectangle = np.transpose(hyperrectangle)

            # Calculate the epsilon for each dimension as ((dim[1] - dim[0]) / (pgd_steps * eps_multiplier))
            eps = []
            for d in hyperrectangle:
                eps.append((d[1] - d[0]) / (pgd_steps * eps_multiplier))
            
            # Generate a pgd point from the hyperrectangle 
            pgd_point = []
            for d in hyperrectangle:
                pgd_point.append(np.random.uniform(d[0], d[1]))
            # PGD attack on the image
            pgd_point = tf.convert_to_tensor([pgd_point], dtype=tf.float32)
            label_0 = tf.convert_to_tensor([[0]], dtype=tf.float32)
            for pgd_step in range(pgd_steps):
                with tf.GradientTape() as tape:
                    tape.watch(pgd_point)
                    prediction = model(pgd_point, training=False)
                    pgd_single_image_loss = pgd_attack_single_image_loss(label_0, prediction)
                # Get the gradients of the loss w.r.t to the input image.
                gradient = tape.gradient(pgd_single_image_loss, pgd_point)
                # Get the sign of the gradients to create the perturbation
                signed_grad = tf.sign(gradient)
                pgd_point = pgd_point + signed_grad * eps
                pgd_point = tf.clip_by_value(pgd_point, t_hyperrectangle[0], t_hyperrectangle[1])
                # print(f"PGD step: {pgd_step + 1}", end="\r")

            # Concatenate the pgd points
            if len(pgd_dataset) > 0:
                pgd_dataset = np.concatenate((pgd_dataset, pgd_point), axis=0)
            else:
                pgd_dataset = pgd_point

        pgd_dataset = np.asarray(pgd_dataset)
        pgd_labels_inside = np.full(len(pgd_dataset), 0)

        # Convert the pgd generated inputs into tf datasets, shuffle and batch them
        pgd_dataset = tf.data.Dataset.from_tensor_slices((pgd_dataset, pgd_labels_inside))
        pgd_dataset = pgd_dataset.shuffle(buffer_size=1024).batch(batch_size)
        return pgd_dataset

def train_adversarial(model, train_dataset, test_dataset, hyperrectangles, epochs, batch_size, n_samples, pgd_steps, alfa=1, beta=1, eps_multiplier=1000, seed=42, from_logits=False):

    optimizer = keras.optimizers.Adam()
    ce_batch_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
    pgd_batch_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
    pgd_attack_single_image_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)

    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    test_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    train_loss_metric = keras.metrics.SparseCategoricalCrossentropy(from_logits=from_logits)
    test_loss_metric = keras.metrics.SparseCategoricalCrossentropy(from_logits=from_logits)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for x_batch_train, y_batch_train in train_dataset:
            # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                outputs = model(x_batch_train, training=True)  # Outputs for this minibatch
                ce_loss_value = ce_batch_loss(y_batch_train, outputs)
                ce_loss_value = ce_loss_value * alfa
            # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(ce_loss_value, model.trainable_weights)
            # Run one step of gradient descent by updating the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        #########################################PGD####################################################
        pgd_dataset = []
        np.random.shuffle(hyperrectangles)
        for hyperrectangle in hyperrectangles[:n_samples]:
            t_hyperrectangle = np.transpose(hyperrectangle)

            # Calculate the epsilon for each dimension as ((dim[1] - dim[0]) / (pgd_steps * eps_multiplier))
            eps = []
            for d in hyperrectangle:
                eps.append((d[1] - d[0]) / (pgd_steps * eps_multiplier))
            
            # Generate a pgd point from the hyperrectangle 
            pgd_point = []
            for d in hyperrectangle:
                pgd_point.append(np.random.uniform(d[0], d[1]))
            # PGD attack on the image
            pgd_point = tf.convert_to_tensor([pgd_point], dtype=tf.float32)
            label_0 = tf.convert_to_tensor([[0]], dtype=tf.float32)
            for pgd_step in range(pgd_steps):
                with tf.GradientTape() as tape:
                    tape.watch(pgd_point)
                    prediction = model(pgd_point, training=False)
                    pgd_single_image_loss = pgd_attack_single_image_loss(label_0, prediction)
                # Get the gradients of the loss w.r.t to the input image.
                gradient = tape.gradient(pgd_single_image_loss, pgd_point)
                # Get the sign of the gradients to create the perturbation
                signed_grad = tf.sign(gradient)
                pgd_point = pgd_point + signed_grad * eps
                pgd_point = tf.clip_by_value(pgd_point, t_hyperrectangle[0], t_hyperrectangle[1])
                # print(f"PGD step: {pgd_step + 1}", end="\r")

            # Concatenate the pgd points
            if len(pgd_dataset) > 0:
                pgd_dataset = np.concatenate((pgd_dataset, pgd_point), axis=0)
            else:
                pgd_dataset = pgd_point

        pgd_dataset = np.asarray(pgd_dataset)
        pgd_labels_inside = np.full(len(pgd_dataset), 0)

        # Convert the pgd generated inputs into tf datasets, shuffle and batch them
        pgd_dataset = tf.data.Dataset.from_tensor_slices((pgd_dataset, pgd_labels_inside))
        pgd_dataset = pgd_dataset.shuffle(buffer_size=1024).batch(batch_size)

        # Iterate over the batches of the pgd dataset.
        for x_batch_train, y_batch_train in pgd_dataset:
            # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                outputs = model(x_batch_train, training=True)  # Outputs for this minibatch
                pgd_loss_value = pgd_batch_loss(y_batch_train, outputs)
                pgd_loss_value = pgd_loss_value * beta
            # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(pgd_loss_value, model.trainable_weights)
            # Run one step of gradient descent by updating the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        ################################################################################################
        
        # Run a training loop at the end of each epoch.
        for x_batch_train, y_batch_train in train_dataset:
            train_outputs = model(x_batch_train, training=False)
            train_acc_metric.update_state(y_batch_train, train_outputs)
            train_loss_metric.update_state(y_batch_train, train_outputs)

        # Run a testing loop at the end of each epoch.
        for x_batch_test, y_batch_test in test_dataset:
            test_outputs = model(x_batch_test, training=False)
            test_acc_metric.update_state(y_batch_test, test_outputs)
            test_loss_metric.update_state(y_batch_test, test_outputs)

        train_acc = train_acc_metric.result()
        test_acc = test_acc_metric.result()
        train_loss = train_loss_metric.result()
        test_loss = test_loss_metric.result()

        train_acc_metric.reset_states()
        test_acc_metric.reset_states()
        train_loss_metric.reset_states()
        test_loss_metric.reset_states()

        print(f"Train acc: {float(train_acc):.4f}, Train loss: {float(train_loss):.4f} --- Test acc: {float(test_acc):.4f}, Test loss: {float(test_loss):.4f} --- Time: {(time.time() - start_time):.2f}s")

    return model
