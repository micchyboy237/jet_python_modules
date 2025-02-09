from functools import lru_cache
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------- FAISS Index Functions ---------------------- #


def estimate_nlist(data_size: int) -> int:
    """Estimate optimal nlist as sqrt(N), ensuring a minimum of 100."""
    return max(100, int(np.sqrt(data_size)))


def estimate_nprobe(nlist: int) -> int:
    """Estimate optimal nprobe as nlist / 10, ensuring a minimum of 1."""
    return max(1, nlist // 10)


def create_faiss_index(data: np.ndarray, dimension: int, nlist: int = None) -> faiss.IndexIVFFlat:
    """
    Creates and trains a FAISS index.

    Parameters:
    - data: np.ndarray, dataset sentence embeddings.
    - dimension: int, embedding dimension.
    - nlist: int, number of clusters for IVF index.

    Returns:
    - index: Trained FAISS IndexIVFFlat.
    """
    data_size = len(data)
    if nlist is None:
        nlist = estimate_nlist(data_size)
    nlist = min(nlist, data_size)

    # Inner Product for cosine similarity
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(
        quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(data)
    index.add(data)

    return index


def search_faiss_index(index: faiss.IndexIVFFlat, query_embedding: np.ndarray, top_k: int, nprobe: int = None):
    """
    Searches FAISS index for nearest neighbors.

    Parameters:
    - index: FAISS IndexIVFFlat.
    - query_embedding: np.ndarray, single query embedding.
    - top_k: int, number of nearest neighbors.
    - nprobe: int, number of clusters to search.

    Returns:
    - distances: np.ndarray, distances to nearest neighbors.
    - indices: np.ndarray, indices of nearest neighbors.
    """
    if nprobe is None:
        nprobe = estimate_nprobe(index.nlist)

    index.nprobe = nprobe
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices


@lru_cache(maxsize=1)
def get_faiss_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')
