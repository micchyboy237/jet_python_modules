import uuid
from typing import List, TypedDict
import numpy as np
from llama_cpp import Llama
from sklearn.preprocessing import normalize
from tqdm import tqdm
from jet.models.tasks.task_types import SimilarityResult, RerankResult


def last_token_pool(embeddings: np.ndarray) -> np.ndarray:
    """Ensure embeddings are returned as-is (already fixed-size)."""
    embeddings = np.array(embeddings)
    return embeddings  # Expect 2D array (batch_size, embedding_dim)


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with task instruction."""
    return f'Instruct: {task_description}\nQuery: {query}'


def encode_with_padding(model: Llama, texts: List[str], max_length: int = 512) -> np.ndarray:
    """Encode texts with padding and return fixed-size embeddings."""
    embeddings = []
    texts_list = tqdm(texts, desc="Encoding") if len(texts) > 1 else texts
    for text in texts_list:
        # Tokenize and pad/truncate to max_length
        tokens = model.tokenize(text.encode('utf-8'), add_bos=True)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + [0] * (max_length - len(tokens))  # Pad with 0
        # Generate embedding
        embedding = model.embed(text)
        embedding = np.array(embedding)
        # Ensure fixed-size embedding (last token if multi-dimensional)
        if len(embedding.shape) > 1:
            embedding = embedding[-1]
        embeddings.append(embedding)
    # Verify shapes before returning
    shapes = [e.shape for e in embeddings]
    if len(set(shapes)) > 1:
        raise ValueError(f"Inconsistent embedding shapes: {shapes}")
    return np.array(embeddings)
