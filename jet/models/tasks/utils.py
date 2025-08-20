import uuid
from typing import List, Optional, TypedDict
import numpy as np
from llama_cpp import Llama
from sklearn.preprocessing import normalize
from tqdm import tqdm
from jet.models.tasks.task_types import SimilarityResult, RerankResult
from jet.models.tokenizer.base import get_tokenizer, count_tokens
from jet.models.tokenizer.utils import calculate_n_ctx
from jet.models.utils import get_embedding_size


def initialize_model(documents: List[str] = [], model_path: Optional[str] = None) -> Llama:
    model_path = model_path or "/Users/jethroestrada/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/8aa0010e73a1075e99dfc213a475a60fd971bbe7/Qwen3-Embedding-0.6B-f16.gguf"
    model_name = "mlx-community/Qwen3-0.6B-4bit-DWQ-053125"

    n_ctx = calculate_n_ctx(model_name, documents)

    settings = {
        "model_path": model_path,
        "embedding": True,
        "n_ctx": n_ctx,
        "n_threads": 4,
        "n_gpu_layers": -1,
        "n_threads_batch": 64,
        "no_perf": True,      # Disable performance timings
        "verbose": True,
        "flash_attn": True,
    }
    return Llama(**settings)


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
