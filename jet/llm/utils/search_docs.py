from jet.llm.utils.transformer_embeddings import generate_embeddings
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import atexit
from jet.llm.mlx.mlx_types import EmbedModelType
from jet.llm.mlx.models import AVAILABLE_EMBED_MODELS, get_embedding_size, resolve_model_key
import numpy as np
from typing import List, Optional, TypedDict, Union, Callable, Tuple
from functools import lru_cache
import logging
from tqdm import tqdm
from jet.logger import logger
import torch
import os
import gc
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader, Dataset


class SimilarityResult(TypedDict):
    """
    Represents a single similarity result for a text.

    Fields:
        id: Identifier for the text.
        rank: Rank based on score (1 for highest).
        doc_index: Original index of the text in the input list.
        score: Normalized similarity score.
        text: The compared text (or chunk if long).
        tokens: Number of encoded tokens from text.
    """
    id: str
    rank: int
    doc_index: int
    score: float
    text: str
    tokens: int


def search_docs(
    query: str,
    documents: List[str],
    model: EmbedModelType = "all-minilm:33m",
    top_k: Optional[int] = 10,
    batch_size: Optional[int] = None,
    normalize: bool = True,
    chunk_size: Optional[int] = None,
    ids: Optional[List[str]] = None,
    threshold: Optional[float] = None
) -> List[SimilarityResult]:
    """Search documents with memory-efficient embedding generation and return SimilarityResult.

    Args:
        query: The query string to search for.
        documents: List of documents to search through.
        model: Embedding model to use (default: "all-minilm:33m").
        top_k: Number of top results to return (default: 10).
        batch_size: Batch size for embedding generation (default: None).
        normalize: Whether to normalize embeddings (default: True).
        chunk_size: Maximum token length for document chunks (default: None).
        ids: Optional list of document IDs (default: None).
        threshold: Minimum similarity score to include in results (default: None).

    Returns:
        List of SimilarityResult objects, sorted by similarity score.
    """
    if not query or not documents:
        raise ValueError("Query string and documents list must not be empty.")

    if not top_k:
        top_k = len(documents)

    # Validate ids if provided
    if ids is not None:
        if len(ids) != len(documents):
            raise ValueError(
                f"Length of ids ({len(ids)}) must match length of documents ({len(documents)})")
        if len(ids) != len(set(ids)):
            raise ValueError("IDs must be unique")

    # Validate threshold
    if threshold is not None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

    # Initialize tokenizer for token counting
    embed_model = resolve_model_key(model)
    model_id = AVAILABLE_EMBED_MODELS[embed_model]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    query_embedding = generate_embeddings(
        model, query, batch_size, normalize, chunk_size=chunk_size)
    doc_embeddings = generate_embeddings(
        model, documents, batch_size, normalize, chunk_size=chunk_size)

    query_embedding = np.array(query_embedding)
    doc_embeddings = np.array(doc_embeddings)

    if len(doc_embeddings) == 0 or len(documents) == 0:
        return []
    if len(doc_embeddings) != len(documents):
        logger.error(
            f"Mismatch between document embeddings ({len(doc_embeddings)}) and documents ({len(documents)})")
        return []

    similarities = np.dot(doc_embeddings, query_embedding) / (
        np.linalg.norm(doc_embeddings, axis=1) *
        np.linalg.norm(query_embedding)
    )

    similarities = np.nan_to_num(similarities, nan=-1.0)

    top_k = min(top_k, len(documents))
    if top_k <= 0:
        return []

    # Apply threshold filtering
    valid_indices = np.where(similarities >= (
        threshold if threshold is not None else -1.0))[0]
    if len(valid_indices) == 0:
        return []

    # Sort by similarity and take top_k
    top_indices = valid_indices[np.argsort(
        similarities[valid_indices])[::-1]][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        doc_text = documents[idx]
        # Count tokens for the document
        tokens = len(tokenizer.encode(doc_text, add_special_tokens=True))
        # Use provided ID if available, otherwise default to f"doc_{idx}"
        doc_id = ids[idx] if ids is not None else f"doc_{idx}"
        result = SimilarityResult(
            id=doc_id,
            rank=rank,
            doc_index=int(idx),
            score=float(similarities[idx]),
            text=doc_text,
            tokens=tokens
        )
        results.append(result)

    return results


def search_docs_with_rerank(
    query: str,
    documents: List[str],
    model: EmbedModelType = "all-minilm:33m",
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: Optional[int] = 10,
    batch_size: Optional[int] = None,
    normalize: bool = True,
    chunk_size: Optional[int] = None,
    ids: Optional[List[str]] = None,
    rerank_top_k: Optional[int] = None
) -> List[SimilarityResult]:
    """
    Search documents with embedding-based similarity and rerank using a cross-encoder.

    Args:
        query: The query string to search for.
        documents: List of document strings to search through.
        model: Embedding model for initial similarity search.
        rerank_model: Cross-encoder model for reranking (e.g., 'cross-encoder/ms-marco-MiniLM-L-6-v2').
        top_k: Number of top results to return after reranking.
        batch_size: Batch size for embedding and reranking.
        normalize: Whether to normalize embeddings.
        chunk_size: Size for chunking large texts.
        ids: Optional list of document IDs.
        rerank_top_k: Number of top results from initial search to rerank (defaults to 2 * top_k).

    Returns:
        List of SimilarityResult dictionaries with reranked scores.
    """
    if not query or not documents:
        raise ValueError("Query string and documents list must not be empty.")

    # Set default rerank_top_k to 2 * top_k or len(documents) if top_k is None
    if top_k is None:
        top_k = len(documents)
    if rerank_top_k is None:
        rerank_top_k = min(2 * top_k, len(documents))
    rerank_top_k = min(rerank_top_k, len(documents))

    # Get initial similarity results
    initial_results = search_docs(
        query=query,
        documents=documents,
        model=model,
        top_k=rerank_top_k,
        batch_size=batch_size,
        normalize=normalize,
        chunk_size=chunk_size,
        ids=ids
    )

    if not initial_results:
        return []

    # Load cross-encoder model
    device = "mps" if torch.backends.mps.is_available(
    ) else "cuda" if torch.cuda.is_available() else "cpu"
    try:
        rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model)
        rerank_model_instance = AutoModelForSequenceClassification.from_pretrained(
            rerank_model, torch_dtype=torch.float16
        ).to(device)
        rerank_model_instance.eval()
    except Exception as e:
        logger.error(f"Failed to load rerank model {rerank_model}: {str(e)}")
        raise RuntimeError(f"Failed to load rerank model: {str(e)}")

    # Prepare query-document pairs for reranking
    pairs = [(query, result["text"]) for result in initial_results]
    if batch_size is None:
        batch_size = 32  # Suitable for Mac M1 MPS

    # Use DataLoader for batching with custom collate_fn
    class PairDataset(Dataset):
        def __init__(self, pairs):
            self.pairs = pairs

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            return self.pairs[idx]

    def collate_fn(batch):
        return batch  # Return batch as a list of tuples

    dataset = PairDataset(pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)

    # Compute reranking scores
    rerank_scores = []
    with torch.autocast(device_type=device, dtype=torch.float16):
        for batch in tqdm(dataloader, desc="Reranking", leave=True):
            # Ensure batch is a list of tuples
            queries, docs = zip(*batch)
            inputs = rerank_tokenizer(
                list(queries),
                list(docs),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = rerank_model_instance(**inputs)
                # Cross-encoder outputs a single score per pair
                scores = outputs.logits.squeeze(-1)
                scores = torch.sigmoid(scores)  # Convert to [0,1] range
                rerank_scores.extend(scores.cpu().tolist())

            del inputs, outputs, scores
            torch.mps.empty_cache()
            gc.collect()

    # Update results with reranking scores
    for i, score in enumerate(rerank_scores):
        initial_results[i]["score"] = float(score)

    # Sort by reranking score and take top_k
    sorted_results = sorted(
        initial_results, key=lambda x: x["score"], reverse=True)
    sorted_results = sorted_results[:top_k]

    # Update ranks
    for rank, result in enumerate(sorted_results, start=1):
        result["rank"] = rank

    return sorted_results
