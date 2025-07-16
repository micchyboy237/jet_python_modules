import os
import shutil
from typing import Optional, TypedDict, List
import numpy as np
from jet.data.header_utils._prepare_for_rag import preprocess_text
from jet.file.utils import load_file, save_file
from jet.models.embeddings.base import generate_embeddings
from jet.models.embeddings.chunking import chunk_headers_by_hierarchy
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.models.tokenizer.base import get_tokenizer_fn
from shared.data_types.job import JobData


class SearchResult(TypedDict):
    rank: int
    score: float
    job_title: str
    content: str


def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
    """Split text into chunks based on token count with overlap."""
    tokenizer = get_tokenizer_fn("mxbai-embed-large")
    tokens = tokenizer(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        start += max_tokens - overlap

    return chunks


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def vector_search(query: str, texts: List[str], embed_model: EmbedModelType, top_k: Optional[int] = None) -> List[SearchResult]:
    """Perform vector search with chunking and return ranked results."""
    # Chunk texts if needed
    chunk_to_doc = []
    for doc_idx, text in enumerate(texts):
        chunk_to_doc.append((doc_idx, text))

    # Preprocess text
    preprocessed_texts = [preprocess_text(text) for text in texts]

    # Generate embeddings for query and all chunks
    embeddings = generate_embeddings(
        [query] + preprocessed_texts,
        embed_model,
        return_format="numpy",
        show_progress=True,
        batch_size=64
    )

    query_embedding = embeddings[0]
    chunk_embeddings = embeddings[1:]

    # Calculate similarities
    similarities = [
        (cosine_similarity(query_embedding, chunk_emb), doc_idx, orig_text)
        for chunk_emb, (doc_idx, orig_text) in zip(chunk_embeddings, chunk_to_doc)
    ]

    # Aggregate scores by document (take max score across chunks)
    doc_scores = {}
    for score, doc_idx, orig_text in similarities:
        if doc_idx not in doc_scores or score > doc_scores[doc_idx][0]:
            doc_scores[doc_idx] = (score, orig_text)

    # Sort by score and create results
    if not top_k:
        top_k = len(texts)
    results = []
    for rank, (doc_idx, (score, content)) in enumerate(
        sorted(doc_scores.items(), key=lambda x: x[1][0], reverse=True)[
            :top_k], 1
    ):
        job_title = content.split("\n\n")[0].replace(
            "# Job Title\n\n", "").strip()
        results.append(SearchResult(
            rank=rank,
            score=float(score),
            job_title=job_title,
            content=content
        ))

    return results
