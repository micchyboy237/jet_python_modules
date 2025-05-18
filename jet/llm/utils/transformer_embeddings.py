from jet.llm.mlx.mlx_types import EmbedModelType
from jet.llm.mlx.models import AVAILABLE_EMBED_MODELS
import numpy as np
from typing import List, Optional, Union, Literal, Callable, Tuple
from functools import lru_cache
import logging
from tqdm import tqdm
from jet.logger import logger
import torch
from transformers import AutoTokenizer, AutoModel


def _calculate_dynamic_batch_size(embedding_dim: int, device: str) -> int:
    """Calculate dynamic batch size based on embedding dimension and device."""
    target_memory = 1024 * 1024 * \
        1024 if device in ["cuda", "mps"] else 512 * 1024 * 1024
    bytes_per_embedding = embedding_dim * 4
    batch_size = int(target_memory / (bytes_per_embedding * 1.2))
    return max(16, min(512, batch_size))


def generate_embeddings(
    model_key: EmbedModelType,
    texts: Union[str, List[str]],
    batch_size: Optional[int] = None,
    normalize: bool = True,
    _model: Optional[AutoModel] = None,
    _tokenizer: Optional[AutoTokenizer] = None,
    use_tqdm: Optional[bool] = None
) -> Union[List[float], List[List[float]]]:
    if not texts:
        raise ValueError("Input texts cannot be empty")

    # Determine device
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # Load model and tokenizer if not provided
    model_id = AVAILABLE_EMBED_MODELS[model_key]
    tokenizer = _tokenizer or AutoTokenizer.from_pretrained(model_id)
    model = _model or AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    embedding_dim = model.config.hidden_size
    if isinstance(texts, str):
        texts = [texts]

    # Validate batch_size
    if batch_size is not None and batch_size <= 0:
        raise ValueError("Batch size must be positive")
    batch_size = batch_size or _calculate_dynamic_batch_size(
        embedding_dim, device)

    all_embeddings = []
    # Determine whether to use tqdm: use_tqdm arg takes precedence, else disable for small inputs
    if use_tqdm is None:
        use_tqdm = len(texts) > 2
    iterator = tqdm(range(0, len(texts), batch_size),
                    desc="Generating embeddings") if use_tqdm else range(0, len(texts), batch_size)
    for i in iterator:
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0).cpu().tolist()
    return all_embeddings[0] if len(texts) == 1 else all_embeddings


def get_embedding_function(
    model_name: str,
    batch_size: Optional[int] = None,
    normalize: bool = True
) -> Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]]:
    """Load a Hugging Face model and tokenizer and return a callable that generates embeddings."""
    logger.info(f"Loading model: {model_name}")
    if model_name not in AVAILABLE_EMBED_MODELS:
        raise ValueError(
            f"Model {model_name} not found in AVAILABLE_EMBED_MODELS. Available models: {list(AVAILABLE_EMBED_MODELS.keys())}")

    model_id = AVAILABLE_EMBED_MODELS[model_name]
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_id}: {str(e)}")

    model.eval()

    def embedding_function(texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        return generate_embeddings(
            model_key=model_name,
            texts=texts,
            batch_size=batch_size,
            normalize=normalize,
            _model=model,
            _tokenizer=tokenizer
        )

    return embedding_function


def search_docs(
    query: str,
    documents: List[str],
    model_key: EmbedModelType,
    top_k: int = 5,
    batch_size: Optional[int] = None,
    normalize: bool = True
) -> List[Tuple[str, float]]:
    """
    Search for documents most relevant to the query using cosine similarity of embeddings.

    Args:
        query: The search query string
        documents: List of documents to search through
        model_key: Model to use for embeddings
        top_k: Number of top resultsto return
        batch_size: Batch size for embedding generation
        normalize: Whether to normalize embeddings

    Returns:
        List of tuples containing (document, similarity_score)
    """
    if not query or not documents:
        return []

    # Generate embeddings
    query_embedding = generate_embeddings(
        model_key, query, batch_size, normalize)
    doc_embeddings = generate_embeddings(
        model_key, documents, batch_size, normalize)

    # Convert to numpy arrays for efficient computation
    query_embedding = np.array(query_embedding)
    doc_embeddings = np.array(doc_embeddings)

    # Compute cosine similarities
    similarities = np.dot(doc_embeddings, query_embedding) / (
        np.linalg.norm(doc_embeddings, axis=1) *
        np.linalg.norm(query_embedding)
    )

    # Get top_k indices and scores
    top_indices = np.argsort(similarities)[::-1][:min(top_k, len(documents))]
    results = [
        (documents[idx], float(similarities[idx]))
        for idx in top_indices
    ]

    return results
