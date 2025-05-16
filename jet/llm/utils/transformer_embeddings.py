import numpy as np
from typing import List, Optional, Union, Literal, Callable
from functools import lru_cache
import logging
from tqdm import tqdm  # Progress tracking
from jet.logger import logger
import torch
from transformers import AutoTokenizer, AutoModel

# Supported model mapping
EMBED_MODELS = {
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5",
    "mxbai-embed-large": "mixedbread-ai/mxbai-embed-large-v1",
    "granite-embedding": "ibm-granite/granite-embedding-30m-english",
    "granite-embedding:278m": "ibm-granite/granite-embedding-278m-multilingual",
    "all-minilm:22m": "sentence-transformers/all-MiniLM-L6-v2",
    "all-minilm:33m": "sentence-transformers/all-MiniLM-L12-v2",
    "snowflake-arctic-embed:33m": "Snowflake/snowflake-arctic-embed-s",
    "snowflake-arctic-embed:137m": "Snowflake/snowflake-arctic-embed-m-long",
    "snowflake-arctic-embed": "Snowflake/snowflake-arctic-embed-l",
    "paraphrase-multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "bge-large": "BAAI/bge-large-en-v1.5",
}


def _calculate_dynamic_batch_size(embedding_dim: int, device: str) -> int:
    """Calculate dynamic batch size based on embedding dimension and device."""
    # Base memory estimate: 4 bytes per float32 * embedding_dim * batch_size
    # Target ~1GB memory usage for GPU, 512MB for CPU/MPS
    target_memory = 1024 * 1024 * \
        1024 if device in ["cuda", "mps"] else 512 * 1024 * 1024
    bytes_per_embedding = embedding_dim * 4  # float32
    # Add 20% overhead for tokenizer and model buffers
    batch_size = int(target_memory / (bytes_per_embedding * 1.2))
    # Clamp batch size between 16 and 512 for practicality
    return max(16, min(512, batch_size))


def generate_embeddings(
    model_key: Literal[*EMBED_MODELS.keys()],
    texts: Union[str, List[str]],
    batch_size: Optional[int] = None,
    normalize: bool = True
) -> Union[List[float], List[List[float]]]:
    """
    Generate embeddings using a selected Hugging Face model with progress tracking.
    """
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model_id = EMBED_MODELS[model_key]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    # Get embedding dimension from model configuration
    embedding_dim = model.config.hidden_size

    if isinstance(texts, str):
        texts = [texts]

    all_embeddings = []
    if batch_size is None:
        batch_size = _calculate_dynamic_batch_size(embedding_dim, device)

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           return_tensors="pt").to(device)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0).cpu().tolist()
    return all_embeddings[0] if len(texts) == 1 else all_embeddings


def get_embedding_function(model_name: str, batch_size: Optional[int] = None, normalize: bool = True) -> Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]]:
    """Load a Hugging Face model and tokenizer and return a callable that generates embeddings."""
    logger.info(f"Loading model: {model_name}")

    if model_name not in EMBED_MODELS:
        raise ValueError(
            f"Model {model_name} not found in EMBED_MODELS. Available models: {list(EMBED_MODELS.keys())}")

    # Load model and tokenizer once
    model_id = EMBED_MODELS[model_name]
    device = "mps" if torch.backends.mps.is_available(
    ) else "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_id}: {str(e)}")
    model.eval()

    def embedding_function(texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for input texts using the specified model.
        """
        # Reuse generate_embeddings with preloaded model and tokenizer
        return generate_embeddings(
            model_key=model_name,
            texts=texts,
            batch_size=batch_size,
            normalize=normalize,
            _model=model,  # Pass preloaded model
            _tokenizer=tokenizer  # Pass preloaded tokenizer
        )

    return embedding_function
