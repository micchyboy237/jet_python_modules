from typing import List, Optional, TypedDict
import numpy as np
from numpy.typing import NDArray
from llama_cpp import Llama
from sklearn.preprocessing import normalize
from tqdm import tqdm


class SearchResult(TypedDict):
    id: str
    doc_index: int
    rank: int
    score: float
    text: str


class VectorSearch:
    """Encapsulates vector search functionality using a language model for embeddings."""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 4, n_gpu_layers: int = 0):
        """
        Initialize the VectorSearch with a Llama model.

        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context length for the model
            n_threads: Number of CPU threads (optimized for Mac M1)
            n_gpu_layers: Number of GPU layers (0 for CPU-only)
        """
        try:
            self.model = Llama(
                model_path=model_path,
                embedding=True,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from {model_path}: {str(e)}")

    def __del__(self):
        """Clean up model resources."""
        if hasattr(self, 'model'):
            self.model.close()

    def encode_texts(self, texts: List[str], max_length: int = 512) -> NDArray[np.float32]:
        """
        Encode texts into embeddings with padding/truncation.

        Args:
            texts: List of texts to encode
            max_length: Maximum token length for encoding

        Returns:
            Array of embeddings (num_texts, embedding_dim)

        Raises:
            ValueError: If embedding shapes are inconsistent
        """
        embeddings = []
        texts_iter = tqdm(texts, desc="Encoding texts") if len(
            texts) > 1 else texts
        for text in texts_iter:
            tokens = self.model.tokenize(text.encode('utf-8'), add_bos=True)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [0] * (max_length - len(tokens))
            embedding = np.array(self.model.embed(text), dtype=np.float32)
            if len(embedding.shape) > 1:
                embedding = embedding[-1]
            embeddings.append(embedding)

        embeddings = np.array(embeddings)
        shapes = [e.shape for e in embeddings]
        if len(set(shapes)) > 1:
            raise ValueError(f"Inconsistent embedding shapes: {shapes}")
        return embeddings

    def normalize_embeddings(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Normalize embeddings using L2 norm.

        Args:
            embeddings: Array of embeddings (num_vectors, embedding_dim)

        Returns:
            Normalized embeddings
        """
        return normalize(embeddings, norm='l2', axis=1)

    def compute_similarity(self, query_embeddings: NDArray[np.float32],
                           doc_embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Compute cosine similarity between query and document embeddings.

        Args:
            query_embeddings: Query embeddings (num_queries, embedding_dim)
            doc_embeddings: Document embeddings (num_docs, embedding_dim)

        Returns:
            Similarity matrix (num_queries, num_docs)
        """
        return query_embeddings @ doc_embeddings.T

    def search(self, queries: List[str], documents: List[dict], top_k: int = 3) -> List[List[SearchResult]]:
        """
        Perform vector search and return structured results.

        Args:
            queries: List of query strings
            documents: List of dictionaries with 'id' and 'text' keys
            top_k: Number of top results to return per query

        Returns:
            List of SearchResult lists, one per query
        """
        # Encode and normalize
        query_embeddings = self.encode_texts(queries)
        doc_texts = [doc['text'] for doc in documents]
        doc_ids = [doc['id'] for doc in documents]
        doc_embeddings = self.encode_texts(doc_texts)
        query_embeddings = self.normalize_embeddings(query_embeddings)
        doc_embeddings = self.normalize_embeddings(doc_embeddings)

        # Compute similarities
        scores = self.compute_similarity(query_embeddings, doc_embeddings)

        # Format results
        results = []
        for query_idx, query_scores in enumerate(scores):
            # Sort by score in descending order
            top_indices = np.argsort(query_scores)[
                ::-1][:min(top_k, len(documents))]
            query_results = []
            for rank, doc_idx in enumerate(top_indices, 1):
                text = documents[doc_idx]['text']
                query_results.append({
                    'id': doc_ids[doc_idx],
                    'doc_index': doc_idx,
                    'rank': rank,
                    'score': float(query_scores[doc_idx]),
                    'text': text
                })
            results.append(query_results)
        return results


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with task instruction."""
    return f'Instruct: {task_description}\nQuery: {query}'
