import logging
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
from jet.adapters.bertopic import BERTopic

def example_cluster_embeddings():
    """Demonstrate clustering embeddings with different clustering models."""
    logging.info("Starting cluster embeddings example...")
    samples, features, centers = 200, 500, 4
    embeddings, _ = make_blobs(n_samples=samples, centers=centers, n_features=features, random_state=42)
    documents = [str(i + 1) for i in range(embeddings.shape[0])]
    old_df = pd.DataFrame({"Document": documents, "ID": range(len(documents)), "Topic": None})
    
    # Test with KMeans
    logging.info("Clustering with KMeans...")
    kmeans_model = KMeans(n_clusters=centers)
    kmeans_bertopic = BERTopic(hdbscan_model=kmeans_model)
    kmeans_df, _ = kmeans_bertopic._cluster_embeddings(embeddings, old_df)
    logging.info(f"KMeans clustering produced {len(kmeans_df.Topic.unique())} unique topics")
    
    # Test with HDBSCAN
    logging.info("Clustering with HDBSCAN...")
    hdbscan_model = HDBSCAN(min_cluster_size=10, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    hdbscan_bertopic = BERTopic(hdbscan_model=hdbscan_model)
    hdbscan_df, _ = hdbscan_bertopic._cluster_embeddings(embeddings, old_df)
    logging.info(f"HDBSCAN clustering produced {len(hdbscan_df.Topic.unique())} unique topics")
    
    return kmeans_df, hdbscan_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_cluster_embeddings()
    logging.info("Cluster usage examples completed successfully.")
