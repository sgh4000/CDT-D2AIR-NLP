from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def visualise(X_pos_train_embed_align, X_neg_train_embed_align, X_pos_test_embed_align, X_neg_test_embed_align, X_pos_strings_train, X_pos_strings_test, X_neg_strings_train, X_neg_strings_test, Y_pos_class_train, Y_pos_class_test, Y_neg_class_train, Y_neg_class_test, X_concat_strings_train, X_concat_strings_test, Y_concat_class_train, Y_concat_class_test, X_pos_train_PCA, X_neg_train_PCA, X_pos_test_PCA, X_neg_test_PCA):
    #bins to look at how much data is in each and to look at balance
    #t-SNE
    #UMAP nearest neighbours
    #cosine
    #silhouette
    return 



def tsne_display(X, y, title):
    tsne = TSNE(n_components=2, random_state=42)
    examples_2d = tsne.fit_transform(X)

    plt.figure(figsize=(10,6))
    plt.scatter(examples_2d[:,0], examples_2d[:,1], c=y[:], cmap='coolwarm')
    plt.colorbar(label='Label')
    plt.title(title)
    plt.savefig(f"data/visualisation/{title}.png", dpi=300, bbox_inches='tight')
    plt.show()
