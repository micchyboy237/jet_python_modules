from collections import Counter
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import Any, List, Iterator, Literal, Optional, Dict, TypedDict
from collections import defaultdict
import hashlib
import uuid
from jet.logger import logger
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_types import ModelType
from jet.models.utils import resolve_model_value
import mlx.core as mx
from mlx_lm import load
from transformers import AutoTokenizer
import numpy as np
import pickle
import os

_model_cache: Dict[str, Any] = {}
_tokenizer_cache: Dict[str, Any] = {}
_persistent_embedding_cache: Dict[str, np.ndarray] = {}
_default_cache_file = "embedding_cache.pkl"
seed = 45


def load_persistent_cache(cache_file: str) -> None:
    """Load persistent embedding cache from file."""
    global _persistent_embedding_cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                _persistent_embedding_cache = pickle.load(f)
            logger.info(
                f"Loaded persistent cache with {len(_persistent_embedding_cache)} entries from {cache_file}")
        except Exception as e:
            logger.error(
                f"Failed to load persistent cache from {cache_file}: {str(e)}")
    else:
        logger.info(
            f"No persistent cache file found at {cache_file}, starting fresh")


def save_persistent_cache(cache_file: str) -> None:
    """Save persistent embedding cache to file."""
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(_persistent_embedding_cache, f)
        logger.info(
            f"Saved persistent cache with {len(_persistent_embedding_cache)} entries to {cache_file}")
    except Exception as e:
        logger.error(
            f"Failed to save persistent cache to {cache_file}: {str(e)}")


def generate_summary(query: str, results: List[Dict], chunks: List[str], total_embed: float, total_classify: float, total_time: float) -> str:
    total_chunks = len(chunks)
    relevant_count = sum(1 for r in results if r["label"] == "relevant")
    non_relevant_count = total_chunks - relevant_count
    relevant_percentage = (relevant_count / total_chunks *
                           100) if total_chunks > 0 else 0
    non_relevant_percentage = 100 - relevant_percentage
    relevant_scores = [r["score"] for r in results if r["label"] == "relevant"]
    avg_relevant_score = sum(relevant_scores) / \
        len(relevant_scores) if relevant_scores else 0
    source_url_counts = Counter(r["source_url"]
                                for r in results if r["label"] == "relevant")
    top_relevant = sorted([r for r in results if r["label"] ==
                          "relevant"], key=lambda x: x["score"], reverse=True)[:3]
    summary = [
        f"**Query**: {query}",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        f"- **Total Chunks Processed**: {total_chunks}",
        f"- **Relevant Chunks**: {relevant_count} ({relevant_percentage:.2f}%)",
        f"- **Non-Relevant Chunks**: {non_relevant_count} ({non_relevant_percentage:.2f}%)",
        f"- **Average Score for Relevant Chunks**: {avg_relevant_score:.4f}",
        "",
        f"- **Embedding Generation Time**: {timedelta(seconds=total_embed)} ({total_embed:.2f} seconds)",
        f"- **Classification Time**: {timedelta(seconds=total_classify)} ({total_classify:.2f} seconds)",
        f"- **Total Execution Time**: {timedelta(seconds=total_time)} ({total_time:.2f} seconds)",
        "",
        "\n".join(f"- {url}: {count} chunk(s)" for url,
                  count in source_url_counts.items()) or "- None",
        "",
    ]
    if top_relevant:
        for i, r in enumerate(top_relevant, 1):
            summary.extend([
                f"**Relevant Chunk {i}**:",
                f"- **Source URL**: {r['source_url']}",
                f"- **Text**: {r['text'][:100]}{'...' if len(r['text']) > 100 else ''}",
                ""
            ])
    else:
        summary.append("- No relevant chunks found.")
    return "\n".join(summary)


class ClassificationResult(TypedDict):
    id: str
    doc_index: int
    rank: int
    score: float
    text: str
    label: Literal["relevant", "non-relevant",
                   "highly relevant", "moderately relevant"]
    threshold: float


class MLXRAGClassifier:
    """Processes preprocessed web data with MLX for RAG usage."""

    def __init__(self, model_name: ModelType = "qwen3-1.7b-4bit", batch_size: int = 4, show_progress: bool = False, use_persistent_cache: bool = False, cache_file_path: str = _default_cache_file):
        """Initialize with MLX model, tokenizer, batch size, progress display option, and persistent cache settings."""
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.query_cache: Dict[str, np.ndarray] = {}
        self.use_persistent_cache = use_persistent_cache
        self.cache_file_path = cache_file_path
        if self.use_persistent_cache:
            load_persistent_cache(self.cache_file_path)
        try:
            model_path = resolve_model_value(model_name)
            if model_name in _model_cache and model_name in _tokenizer_cache:
                self.model = _model_cache[model_name]
                self.tokenizer = _tokenizer_cache[model_name]
                logger.info(
                    f"Reused cached model and tokenizer for {model_name}")
            else:
                self.model = MLXModelRegistry.load_model(model_path, seed=seed)
                self.tokenizer = MLXModelRegistry.get_tokenizer(model_path)
                _model_cache[model_name] = self.model
                _tokenizer_cache[model_name] = self.tokenizer
                logger.info(
                    f"Loaded and cached model and tokenizer for {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {str(e)}")
            raise

    def _hash_text(self, text: str) -> str:
        """Generate a hash for a given text string."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate and cache query embedding."""
        query_hash = self._hash_text(query)
        if query_hash in self.query_cache:
            logger.debug(
                f"Retrieved query embedding from cache for hash: {query_hash}")
            return self.query_cache[query_hash]
        if not query.strip():
            logger.error("Empty query provided")
            raise ValueError("Query cannot be empty")
        query_inputs = self.tokenizer(
            query, return_tensors="np", padding=True, truncation=True, max_length=512)
        query_input_ids = mx.array(query_inputs["input_ids"]).astype(mx.int32)
        query_output = self.model(query_input_ids)
        query_embedding = np.array(
            mx.mean(query_output, axis=1).tolist(), dtype=np.float32).squeeze()
        if np.all(query_embedding == 0):
            logger.error("Query embedding is a zero vector")
            raise ValueError("Invalid query embedding: zero vector")
        self.query_cache[query_hash] = query_embedding
        logger.debug(f"Cached query embedding for hash: {query_hash}")
        return query_embedding

    def generate_embeddings(self, chunks: List[str], group_ids: Optional[List[str]] = None) -> np.ndarray:
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        if not chunks:
            logger.warning("Empty chunk list provided, returning empty array")
            return np.array([])
        embeddings = []
        cached_count = 0
        chunk_to_indices = defaultdict(list)
        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                logger.warning(f"Skipping empty chunk at index {idx}")
                continue
            chunk_to_indices[chunk].append(idx)
        unique_chunks = list(chunk_to_indices.keys())
        chunk_index_map = {chunk: indices for chunk,
                           indices in chunk_to_indices.items()}
        final_embeddings = [None] * len(chunks)
        for chunk, indices in chunk_index_map.items():
            chunk_hash = self._hash_text(chunk)
            if self.use_persistent_cache and chunk_hash in _persistent_embedding_cache:
                emb = _persistent_embedding_cache[chunk_hash]
                cached_count += len(indices)
                for idx in indices:
                    final_embeddings[idx] = emb
                if chunk == "#### Louis Bouchard":
                    logger.debug(
                        f"Retrieved cached embedding for '#### Louis Bouchard' at indices {indices}: {emb[:5]}...")
                continue
            # Tokenize individually to avoid batch padding effects
            inputs = self.tokenizer(
                chunk, return_tensors="np", padding=True, truncation=True, max_length=512)
            input_ids = mx.array(inputs["input_ids"]).astype(mx.int32)
            if chunk == "#### Louis Bouchard":
                logger.debug(
                    f"Tokenized '#### Louis Bouchard': input_ids={inputs['input_ids'][0][:10]}...")
            output = self.model(input_ids)
            emb = np.array(mx.mean(output, axis=1).tolist(),
                           dtype=np.float32).squeeze()
            if np.all(emb == 0):
                logger.warning(f"Zero-vector embedding for chunk: {chunk}")
                continue
            if self.use_persistent_cache:
                _persistent_embedding_cache[chunk_hash] = emb
            for idx in indices:
                final_embeddings[idx] = emb
            if chunk == "#### Louis Bouchard":
                logger.debug(
                    f"Generated embedding for '#### Louis Bouchard' at indices {indices}: {emb[:5]}...")
        if self.use_persistent_cache:
            save_persistent_cache(self.cache_file_path)
        embeddings_array = np.stack(final_embeddings)
        if embeddings_array.size > 0 and np.any(np.linalg.norm(embeddings_array, axis=1) == 0):
            logger.error("Zero-vector embeddings detected in final array")
            raise ValueError("Invalid embeddings: zero vectors detected")
        logger.info(
            f"Generated embeddings shape: {embeddings_array.shape}, {cached_count} chunks retrieved from cache")
        return embeddings_array

    def generate(self, query: str, chunks: List[str], embeddings: np.ndarray, relevance_threshold: float = 0.7) -> Literal["relevant", "non-relevant"]:
        """Classify if the most relevant chunk is relevant or non-relevant based on query similarity."""
        logger.info(
            f"Classifying query: {query}, threshold: {relevance_threshold}")
        if not chunks or embeddings.shape[0] != len(chunks):
            logger.error(
                "Invalid chunks or embeddings, returning non-relevant")
            return "non-relevant"
        query_embedding = self._generate_query_embedding(query)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_embeddings = embeddings / norms
        norm_query = query_embedding / np.linalg.norm(query_embedding)
        similarities = np.dot(norm_embeddings, norm_query)
        # Ensure similarities are in [0, 1]
        similarities = np.clip(similarities, 0, 1)
        top_idx = np.argmax(similarities)
        similarity_score = similarities[top_idx]
        label: Literal["relevant",
                       "non-relevant"] = "relevant" if similarity_score >= relevance_threshold else "non-relevant"
        logger.info(f"Query classified as {label}")
        return label

    def stream_generate(self, query: str, chunks: List[str], embeddings: np.ndarray, top_k: int = 3, relevance_threshold: float = 0.7) -> Iterator[tuple[Literal["relevant", "non-relevant"], float, int]]:
        """Stream classification labels, scores, and indices for top-k chunks based on query similarity."""
        logger.info(
            f"Streaming classifications for query: {query}, top_k: {top_k}, threshold: {relevance_threshold}")
        if top_k < 1:
            logger.warning("top_k must be at least 1, setting to 1")
            top_k = 1
        if not chunks or embeddings.shape[0] != len(chunks):
            logger.error("Invalid chunks or embeddings, cannot stream")
            return
        query_embedding = self._generate_query_embedding(query)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_embeddings = embeddings / norms
        norm_query = query_embedding / np.linalg.norm(query_embedding)
        similarities = np.dot(norm_embeddings, norm_query)
        # Ensure similarities are in [0, 1]
        similarities = np.clip(similarities, 0, 1)
        for idx, score in enumerate(similarities):
            logger.debug(
                f"Chunk {idx}: '{chunks[idx]}' - Similarity: {score:.4f}")
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        for idx in top_indices:
            similarity_score = similarities[idx]
            label: Literal["relevant",
                           "non-relevant"] = "relevant" if similarity_score >= relevance_threshold else "non-relevant"
            logger.debug(
                f"Yielding: label={label}, score={similarity_score:.4f}, index={idx}")
            yield label, similarity_score, idx
        logger.info("Streaming classification completed successfully")

    def classify(self, query: str, chunks: List[str], embeddings: np.ndarray, ids: Optional[List[str]] = None, verbose: bool = False, relevance_threshold: float = 0.7) -> List[ClassificationResult]:
        logger.info(
            f"Classifying query: {query}, {len(chunks)} chunks, verbose={verbose}, threshold={relevance_threshold}")
        if not chunks or embeddings.shape[0] != len(chunks):
            logger.error("Invalid chunks or embeddings, returning empty list")
            return []
        if ids and len(ids) != len(chunks):
            logger.warning(
                "Length of ids does not match chunks, generating UUIDs")
            ids = None
        if not ids:
            ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        query_embedding = self._generate_query_embedding(query)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_embeddings = embeddings / norms
        norm_query = query_embedding / np.linalg.norm(query_embedding)
        results: List[ClassificationResult] = []
        for idx, (chunk, chunk_id, norm_embedding) in enumerate(zip(chunks, ids, norm_embeddings)):
            if np.all(norm_embedding == 0):
                logger.warning(
                    f"Zero-vector embedding for chunk at index {idx}, skipping")
                score = 0.0
            else:
                score = float(np.dot(norm_embedding, norm_query))
                score = np.clip(score, 0, 1)  # Ensure score is in [0, 1]
            label: Literal["relevant",
                           "non-relevant"] = "relevant" if score >= relevance_threshold else "non-relevant"
            result: ClassificationResult = {
                "id": chunk_id,
                "doc_index": idx,
                "rank": 0,
                "score": score,
                "text": chunk,
                "label": label,
                "threshold": relevance_threshold
            }
            results.append(result)
            if verbose:
                logger.debug(
                    f"Chunk {idx}: id={chunk_id}, score={score:.4f}, label={label}, threshold={relevance_threshold}, text='{chunk[:50]}...'")
        sorted_results = sorted(
            results, key=lambda x: x["score"], reverse=True)
        for rank, result in enumerate(sorted_results, start=1):
            result["rank"] = rank
        logger.info(f"Classification completed, {len(sorted_results)} results")
        return sorted_results

    def classify_multi_label(self, query: str, chunks: List[str], embeddings: np.ndarray, ids: Optional[List[str]] = None, verbose: bool = False, high_threshold: float = 0.9, moderate_threshold: float = 0.7) -> List[ClassificationResult]:
        """Classify chunks into 'highly relevant', 'moderately relevant', or 'non-relevant' based on query similarity."""
        logger.info(
            f"Classifying query (multi-label): {query}, {len(chunks)} chunks, verbose={verbose}, high_threshold={high_threshold}, moderate_threshold={moderate_threshold}")
        if high_threshold <= moderate_threshold:
            logger.error(
                "high_threshold must be greater than moderate_threshold")
            raise ValueError(
                "high_threshold must be greater than moderate_threshold")
        if not chunks or embeddings.shape[0] != len(chunks):
            logger.error("Invalid chunks or embeddings, returning empty list")
            return []
        if ids and len(ids) != len(chunks):
            logger.warning(
                "Length of ids does not match chunks, generating UUIDs")
            ids = None
        if not ids:
            ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        query_embedding = self._generate_query_embedding(query)
        norm_query = query_embedding / np.linalg.norm(query_embedding)
        results: List[ClassificationResult] = []
        for idx, (chunk, chunk_id, embedding) in enumerate(zip(chunks, ids, embeddings)):
            norm_embedding = embedding / np.linalg.norm(embedding)
            if np.all(norm_embedding == 0):
                logger.warning(
                    f"Zero-vector embedding for chunk at index {idx}, skipping")
                score = 0.0
            else:
                score = float(np.dot(norm_embedding, norm_query))
                score = np.clip(score, 0, 1)  # Ensure score is in [0, 1]
            if score >= high_threshold:
                label: Literal["highly relevant", "moderately relevant",
                               "non-relevant"] = "highly relevant"
            elif score >= moderate_threshold:
                label = "moderately relevant"
            else:
                label = "non-relevant"
            result: ClassificationResult = {
                "id": chunk_id,
                "doc_index": idx,
                "rank": 0,
                "score": score,
                "text": chunk,
                "label": label,
                "threshold": moderate_threshold
            }
            results.append(result)
            if verbose:
                logger.debug(
                    f"Chunk {idx}: id={chunk_id}, score={score:.4f}, label={label}, high_threshold={high_threshold}, moderate_threshold={moderate_threshold}, text='{chunk[:50]}...'")
        sorted_results = sorted(
            results, key=lambda x: x["score"], reverse=True)
        for rank, result in enumerate(sorted_results, start=1):
            result["rank"] = rank
        logger.info(
            f"Multi-label classification completed, {len(sorted_results)} results")
        return sorted_results

    def clear_cache(self) -> None:
        """Clear the embedding, query, model, and tokenizer caches."""
        logger.info("Clearing embedding, query, model, and tokenizer caches")
        self.embedding_cache.clear()
        self.query_cache.clear()
        _model_cache.clear()
        _tokenizer_cache.clear()
        if self.use_persistent_cache:
            global _persistent_embedding_cache
            _persistent_embedding_cache.clear()
            if os.path.exists(self.cache_file_path):
                os.remove(self.cache_file_path)
        logger.info("All caches cleared successfully")
