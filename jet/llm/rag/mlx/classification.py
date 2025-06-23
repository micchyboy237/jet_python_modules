from tqdm import tqdm
from typing import Any, List, Iterator, Literal, Optional, Dict, TypedDict
from collections import defaultdict
import hashlib
import uuid
from jet.logger import logger
from jet.models.model_types import ModelType
from jet.models.utils import resolve_model_value
import mlx.core as mx
from mlx_lm import load
from transformers import AutoTokenizer
import numpy as np

_model_cache: Dict[str, Any] = {}
_tokenizer_cache: Dict[str, Any] = {}


class ClassificationResult(TypedDict):
    id: str
    doc_index: int
    rank: int
    score: float
    text: str
    label: Literal["relevant", "non-relevant"]
    threshold: float


class MLXRAGClassifier:
    """Processes preprocessed web data with MLX for RAG usage."""

    def __init__(self, model_name: ModelType = "qwen3-1.7b-4bit", batch_size: int = 4, show_progress: bool = False):
        """Initialize with MLX model, tokenizer, batch size, and progress display option."""
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.query_cache: Dict[str, np.ndarray] = {}
        try:
            model_path = resolve_model_value(model_name)
            if model_name in _model_cache and model_name in _tokenizer_cache:
                self.model = _model_cache[model_name]
                self.tokenizer = _tokenizer_cache[model_name]
                logger.info(
                    f"Reused cached model and tokenizer for {model_name}")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model, _ = load(model_path)
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

    def generate_embeddings(self, chunks: List[str], group_ids: Optional[List[str]] = None) -> np.ndarray:
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = []
        original_indices = list(range(len(chunks)))
        cached_count = 0
        chunk_to_indices = defaultdict(list)
        for idx, chunk in enumerate(chunks):
            chunk_to_indices[chunk].append(idx)
        unique_chunks = list(chunk_to_indices.keys())
        chunk_index_map = {chunk: indices for chunk,
                           indices in chunk_to_indices.items()}
        if group_ids and len(group_ids) == len(chunks):
            url_to_unique_chunks = defaultdict(list)
            for chunk in unique_chunks:
                indices = chunk_index_map[chunk]
                group_id = group_ids[indices[0]]
                url_to_unique_chunks[group_id].append(chunk)
            batches = []
            for url, url_chunks in url_to_unique_chunks.items():
                url_indices = [chunk_index_map[chunk][0]
                               for chunk in url_chunks]
                for i in range(0, len(url_chunks), self.batch_size):
                    batch_chunks = url_chunks[i:i + self.batch_size]
                    batch_indices = url_indices[i:i + self.batch_size]
                    batches.append((batch_chunks, batch_indices))
        else:
            batches = [(unique_chunks[i:i + self.batch_size], original_indices[i:i + self.batch_size])
                       for i in range(0, len(unique_chunks), self.batch_size)]
        num_batches = len(batches)
        iterator = range(num_batches)
        if self.show_progress:
            iterator = tqdm(iterator, total=num_batches,
                            desc="Processing batches")
        for i in iterator:
            batch_chunks, batch_indices = batches[i]
            batch_embeddings = []
            batch_remaining_chunks = []
            batch_remaining_indices = []
            for chunk, idx in zip(batch_chunks, batch_indices):
                chunk_hash = self._hash_text(chunk)
                if chunk_hash in self.embedding_cache:
                    batch_embeddings.append(
                        (idx, self.embedding_cache[chunk_hash]))
                    cached_count += len(chunk_index_map[chunk])
                else:
                    batch_remaining_chunks.append(chunk)
                    batch_remaining_indices.append(idx)
            if batch_remaining_chunks:
                encoded_inputs = [self.tokenizer(
                    chunk, truncation=False, add_special_tokens=True) for chunk in batch_remaining_chunks]
                max_len = max(len(enc["input_ids"]) for enc in encoded_inputs)
                inputs = self.tokenizer(
                    batch_remaining_chunks,
                    return_tensors="np",
                    padding="max_length",
                    truncation=True,
                    max_length=max_len
                )
                input_ids = mx.array(inputs["input_ids"]).astype(mx.int32)
                output = self.model(input_ids)
                embedding = np.array(
                    mx.mean(output, axis=1).tolist(), dtype=np.float32)
                for emb, chunk, idx in zip(embedding, batch_remaining_chunks, batch_remaining_indices):
                    chunk_hash = self._hash_text(chunk)
                    self.embedding_cache[chunk_hash] = emb
                    batch_embeddings.append((idx, emb))
                    cached_count += len(chunk_index_map[chunk]) - 1
                del input_ids, output
                mx.clear_cache()
            embeddings.extend(batch_embeddings)
        final_embeddings = [None] * len(chunks)
        for chunk, indices in chunk_index_map.items():
            chunk_hash = self._hash_text(chunk)
            emb = self.embedding_cache.get(chunk_hash)
            if emb is not None:
                for idx in indices:
                    final_embeddings[idx] = emb
            else:
                logger.error(f"No embedding found for chunk: {chunk}")
                raise ValueError(f"Embedding missing for chunk: {chunk}")
        embeddings_array = np.stack(final_embeddings)
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
        query_hash = self._hash_text(query)
        if query_hash in self.query_cache:
            query_embedding = self.query_cache[query_hash]
        else:
            query_inputs = self.tokenizer(
                query, return_tensors="np", padding=True, truncation=True, max_length=512
            )
            query_input_ids = mx.array(
                query_inputs["input_ids"]).astype(mx.int32)
            query_output = self.model(query_input_ids)
            query_embedding = np.array(
                mx.mean(query_output, axis=1).tolist(), dtype=np.float32).squeeze()
            self.query_cache[query_hash] = query_embedding
        norm_embeddings = embeddings / \
            np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_query = query_embedding / np.linalg.norm(query_embedding)
        similarities = np.dot(norm_embeddings, norm_query)
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
        query_hash = self._hash_text(query)
        if query_hash in self.query_cache:
            query_embedding = self.query_cache[query_hash]
        else:
            query_inputs = self.tokenizer(
                query, return_tensors="np", padding=True, truncation=True, max_length=512
            )
            query_input_ids = mx.array(
                query_inputs["input_ids"]).astype(mx.int32)
            query_output = self.model(query_input_ids)
            query_embedding = np.array(
                mx.mean(query_output, axis=1).tolist(), dtype=np.float32).squeeze()
            self.query_cache[query_hash] = query_embedding
        norm_embeddings = embeddings / \
            np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_query = query_embedding / np.linalg.norm(query_embedding)
        similarities = np.dot(norm_embeddings, norm_query)
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
        """
        Classify chunks based on query similarity, returning a sorted list of ClassificationResult.

        Args:
            query: The query string to classify against.
            chunks: List of text chunks to classify.
            embeddings: Precomputed embeddings for the chunks.
            ids: Optional list of IDs corresponding to chunks. If None, UUIDs are generated.
            verbose: If True, log detailed classification information.
            relevance_threshold: Score threshold for labeling as "relevant" (default: 0.7).

        Returns:
            List of ClassificationResult dictionaries sorted by score in descending order.
        """
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
        query_hash = self._hash_text(query)
        if query_hash in self.query_cache:
            query_embedding = self.query_cache[query_hash]
        else:
            query_inputs = self.tokenizer(
                query, return_tensors="np", padding=True, truncation=True, max_length=512
            )
            query_input_ids = mx.array(
                query_inputs["input_ids"]).astype(mx.int32)
            query_output = self.model(query_input_ids)
            query_embedding = np.array(
                mx.mean(query_output, axis=1).tolist(), dtype=np.float32).squeeze()
            self.query_cache[query_hash] = query_embedding
        norm_embeddings = embeddings / \
            np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_query = query_embedding / np.linalg.norm(query_embedding)
        similarities = np.dot(norm_embeddings, norm_query)
        results: List[ClassificationResult] = []
        for idx, (chunk, chunk_id, score) in enumerate(zip(chunks, ids, similarities)):
            label: Literal["relevant",
                           "non-relevant"] = "relevant" if score >= relevance_threshold else "non-relevant"
            result: ClassificationResult = {
                "id": chunk_id,
                "doc_index": idx,
                "rank": 0,
                "score": float(score),
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
