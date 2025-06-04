import matplotlib.pyplot as plt
import umap
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def visualize_clusters(texts, model_name="intfloat/e5-base-v2", n_clusters=2):
    model = SentenceTransformer(model_name)
    passages = [f"passage: {t}" for t in texts]
    embeddings = model.encode(passages, normalize_embeddings=True)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(embeddings)

    reducer = umap.UMAP(random_state=42)
    reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 6))
    for cluster_id in np.unique(cluster_ids):
        idx = cluster_ids == cluster_id
        plt.scatter(reduced[idx, 0], reduced[idx, 1],
                    label=f'Cluster {cluster_id}')
    for i, text in enumerate(texts):
        plt.annotate(str(i+1), (reduced[i, 0], reduced[i, 1]))
    plt.legend()
    plt.title("2D Visualization of Clusters")
    plt.xlabel("UMAP Dim 1")
    plt.ylabel("UMAP Dim 2")
    plt.tight_layout()
    plt.show()
