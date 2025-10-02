from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import umap
import numpy as np

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

def UMAP_display(X, y, title):
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_emb_2d = reducer.fit_transform(X)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_emb_2d[:, 0], X_emb_2d[:, 1], c=y, cmap='coolwarm', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Labels")
    plt.title(title)
    plt.savefig(f"data/visualisation/{title}.png", dpi=300, bbox_inches='tight')
    plt.show()

def data_balance_display(Xtnp, Xtnn, Xtsp, Xtsn):
    labels = ['Medical Query Train', 'Non-medical Query Train', 'Medical Query Test', 'Non-medical Query Test']
    sizes = [len(Xtnp), len(Xtnn), len(Xtsp), len(Xtsn)]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, sizes)
    title = "Number of Samples Within Each Category to Show Data Balance"
    plt.title(title)
    plt.ylabel("Number of Samples")
    plt.savefig(f"data/visualisation/{title}.png", dpi=300, bbox_inches='tight')
    plt.show()



def print_cosine_sim_demo(Xembed, Xstring, Yclass):
    n_queries = 3
    random_indices = np.random.choice(len(Xembed), size=n_queries, replace=False)

    for query_idx in random_indices:
        query_vec = Xembed[query_idx]

        # Compute cosine similarity with all embeddings
        sim_scores = cosine_similarity([query_vec], Xembed)[0]

        # Get top 5 most similar sentences (excluding the query itself)
        closest_idx = np.argsort(sim_scores)[-6:]  
        closest_idx = closest_idx[closest_idx != query_idx]  # remove the query itself

        # Print results
        print(f"\nQuery sentence: {Xstring[query_idx]}")
        print("Top similar sentences:")
        for idx in closest_idx[::-1]:
            print(f"- {Xstring[idx]} (label={Yclass[idx]}, similarity={sim_scores[idx]:.3f})")