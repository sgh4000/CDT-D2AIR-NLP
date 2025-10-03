from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import umap
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

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

def UMAP_display(Xe, y, title):
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_emb_2d = reducer.fit_transform(Xe)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_emb_2d[:, 0], X_emb_2d[:, 1], c=y, cmap='coolwarm', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Labels")
    plt.title(title)
    plt.savefig(f"data/visualisation/{title}.png", dpi=300, bbox_inches='tight')
   
    
    plt.show()
    return X_emb_2d

def UMAP_investigate(X_emb_2d, Xs, y):
         # --- Create DataFrame with coordinates and labels ---
    umap_df = pd.DataFrame(X_emb_2d, columns=['UMAP1', 'UMAP2'])
    umap_df['label'] = y

    # --- Cluster the UMAP points ---
    kmeans = KMeans(n_clusters=2, random_state=42)
    umap_df['cluster'] = kmeans.fit_predict(X_emb_2d)
    # --- Find misclustered points (label != cluster) ---
    misclustered = umap_df[umap_df['label'] != umap_df['cluster']]

    print(f"Number of misclustered points: {len(misclustered)}\n")

    # --- Inspect the misclustered sentences ---
    mis_idx = misclustered.index
    for i in mis_idx[:20]:  # show first 10 for brevity
        print(f"Label: {y[i]}, Cluster: {umap_df.loc[i, 'cluster']}")
        print(f"Sentence: {Xs[i]}")
        print("---")


def data_balance_display(Xtnp, Xtnn, Xtsp, Xtsn):
    #plot each of the data selected on a bar chart and display number of samples for each
    labels = ['Medical Query Train', 'Non-medical Query Train', 'Medical Query Test', 'Non-medical Query Test']
    sizes = [len(Xtnp), len(Xtnn), len(Xtsp), len(Xtsn)]
    colours = ["#7D2E7C", "#B75EB4", "#3E678E", '#64B5F6'] 
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, sizes, color = colours)
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, str(height),
                 ha='center', va='bottom', fontsize=10)
    title = "Number of Samples Within Each Category to Show Data Balance"
    plt.title(title)
    plt.ylabel("Number of Samples")
    plt.savefig(f"data/visualisation/{title}.png", dpi=300, bbox_inches='tight')
    plt.show()

def print_cosine_sim_demo(Xembed, Xstring, Yclass):
    #given the low silhouette score, it would be nice to probe how the cosine similarity metric works
    #if we take 3 random queries, and find the five 'most similar' strings and see if it is way off as a sanity check
    n_queries = 3
    random_indices = np.random.choice(len(Xembed), size=n_queries, replace=False)
    
    #for each random index
    for query_idx in random_indices:
        #find associated string
        query_vec = Xembed[query_idx]

        # Compute cosine similarity with all embeddings
        sim_scores = cosine_similarity([query_vec], Xembed)[0]

        # Get top 5 most similar sentences (excluding the query itself)
        closest_idx = np.argsort(sim_scores)[-6:]  
        closest_idx = closest_idx[closest_idx != query_idx]  # remove the query itself

        # Print results
        print(f"\nQuery sentence: {Xstring[query_idx]} (label={Yclass[query_idx]})")
        print("Top similar sentences:")
        for idx in closest_idx[::-1]:
            print(f"- {Xstring[idx]} (label={Yclass[idx]}, similarity={sim_scores[idx]:.3f})")

def print_silhouette_score(X):
    #cluster the data in high dimensional space - in this case trying to do so in 2 clusters for the amount of classes we have
    kmeans_hd = KMeans(n_clusters=2, random_state=42)
    #tries to fit each sample in data to the cluster it thinks it is in - will provide output of 0 or 1
    clusters_hd = kmeans_hd.fit_predict(X)

    #computes silhouette score to see how well the data is clustered
    #gives value between -1 and 1: higher values are well matched to own cluster and far from others
    #0 indicates some samples are on or near boundary
    #-1 indicates some samples are in opposite cluster
    score = silhouette_score(X, clusters_hd, metric='cosine')  # cosine rather than euclidean is better for embeddings
    print(f"Silhouette Score: {score:.4f}")