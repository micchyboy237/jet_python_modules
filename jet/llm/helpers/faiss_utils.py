import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def create_faiss_index(data: np.ndarray, dimension: int, nlist: int, metric: int = faiss.METRIC_INNER_PRODUCT) -> faiss.IndexIVFFlat:
    """
    Creates and trains a FAISS index with the given data.

    Parameters:
    - data: np.ndarray, the dataset to index.
    - dimension: int, the dimension of the data.
    - nlist: int, the number of clusters (nlist) for IVF index.
    - metric: FAISS metric type (default: faiss.METRIC_INNER_PRODUCT for cosine similarity).

    Returns:
    - index: Trained FAISS index of type IndexIVFFlat.
    """

    # Ensure nlist is not larger than the number of training points
    nlist = min(nlist, len(data))

    # Flat (brute-force) index for clustering
    quantizer = faiss.IndexFlatIP(dimension)  # Use cosine similarity
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, metric)

    # Train the index
    index.train(data)
    index.add(data)

    return index


def search_faiss_index(index: faiss.IndexIVFFlat, queries: np.ndarray, top_k: int, nprobe: int = 1) -> [np.ndarray, np.ndarray]:
    """
    Searches the FAISS index for the nearest neighbors of the queries.

    Parameters:
    - index: FAISS index.
    - queries: np.ndarray, the query points.
    - top_k: int, the number of nearest neighbors to retrieve.
    - nprobe: int, the number of clusters to search (default: 1).

    Returns:
    - distances: np.ndarray, distances to the nearest neighbors.
    - indices: np.ndarray, indices of the nearest neighbors.
    """
    index.nprobe = nprobe
    distances, indices = index.search(queries, top_k)

    # Convert inner product to cosine similarity score (for better interpretation)
    distances = distances / np.linalg.norm(queries, axis=1)[:, None]
    return distances, indices


def get_model():
    # return SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return SentenceTransformer('all-MiniLM-L6-v2')
