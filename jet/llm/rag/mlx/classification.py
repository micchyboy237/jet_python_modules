from tqdm import tqdm
from typing import List, Iterator, Literal
from jet.logger import logger
from jet.models.model_types import ModelType
from jet.models.utils import resolve_model_value
import mlx.core as mx
from mlx_lm import load
from transformers import AutoTokenizer
import numpy as np


class MLXRAGClassifier:
    """Processes preprocessed web data with MLX for RAG usage."""

    def __init__(self, model_name: ModelType = "qwen3-1.7b-4bit", batch_size: int = 4, show_progress: bool = False):
        """Initialize with MLX model, tokenizer, batch size, and progress display option."""
        logger.debug(
            f"Loading MLX model: {model_name}, batch_size: {batch_size}, show_progress: {show_progress}")
        self.batch_size = batch_size
        self.show_progress = show_progress
        try:
            model_path = resolve_model_value(model_name)
            logger.debug(f"Resolved model path: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model, _ = load(model_path)
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {str(e)}")
            raise

    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks using MLX with batch processing."""
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = []
        num_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        iterator = range(0, len(chunks), self.batch_size)
        if self.show_progress:
            iterator = tqdm(iterator, total=num_batches,
                            desc="Processing batches")

        for i in iterator:
            batch_chunks = chunks[i:i + self.batch_size]
            logger.debug(
                f"Processing batch {i//self.batch_size + 1} with {len(batch_chunks)} chunks")

            # Tokenize individually to compute actual token lengths
            encoded_inputs = [self.tokenizer(
                chunk, truncation=False, add_special_tokens=True) for chunk in batch_chunks]
            max_len = max(len(enc["input_ids"]) for enc in encoded_inputs)
            logger.debug(f"Dynamic max_length for this batch: {max_len}")

            # Re-tokenize with consistent padding/truncation using max_len
            inputs = self.tokenizer(
                batch_chunks,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=max_len
            )

            input_ids = mx.array(inputs["input_ids"]).astype(mx.int32)
            logger.debug(
                f"Batch input IDs shape: {input_ids.shape}, dtype: {input_ids.dtype}")
            output = self.model(input_ids)
            logger.debug(
                f"Batch output shape: {output.shape}, dtype: {output.dtype}")

            embedding = np.array(
                mx.mean(output, axis=1).tolist(), dtype=np.float32)
            logger.debug(
                f"Batch NumPy embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
            embeddings.extend(embedding)

            del input_ids, output
            mx.clear_cache()

        embeddings_array = np.stack(embeddings)
        logger.info(f"Generated embeddings shape: {embeddings_array.shape}")
        return embeddings_array

    def generate(self, query: str, chunks: List[str], embeddings: np.ndarray, relevance_threshold: float = 0.7) -> Literal["relevant", "non-relevant"]:
        """Classify if the most relevant chunk is relevant or non-relevant based on query similarity."""
        logger.info(
            f"Classifying query: {query}, threshold: {relevance_threshold}")
        if not chunks or embeddings.shape[0] != len(chunks):
            logger.error(
                "Invalid chunks or embeddings, returning non-relevant")
            return "non-relevant"

        query_inputs = self.tokenizer(
            query, return_tensors="np", padding=True, truncation=True, max_length=512
        )
        logger.debug(
            f"Query input IDs shape: {query_inputs['input_ids'].shape}, dtype: {query_inputs['input_ids'].dtype}")
        query_input_ids = mx.array(query_inputs["input_ids"]).astype(mx.int32)
        query_output = self.model(query_input_ids)
        logger.debug(
            f"Query output shape: {query_output.shape}, dtype: {query_output.dtype}")
        query_embedding = np.array(
            mx.mean(query_output, axis=1).tolist(), dtype=np.float32).squeeze()
        logger.debug(
            f"Query embedding shape: {query_embedding.shape}, dtype: {query_embedding.dtype}")

        norm_embeddings = embeddings / \
            np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_query = query_embedding / np.linalg.norm(query_embedding)
        logger.debug(
            f"Norm query shape: {norm_query.shape}, Norm embeddings shape: {norm_embeddings.shape}")
        similarities = np.dot(norm_embeddings, norm_query)
        logger.debug(f"Similarities shape: {similarities.shape}")
        top_idx = np.argmax(similarities)
        similarity_score = similarities[top_idx]

        label: Literal["relevant",
                       "non-relevant"] = "relevant" if similarity_score >= relevance_threshold else "non-relevant"
        logger.debug(
            f"Top chunk index: {top_idx}, score: {similarity_score:.4f}, label: {label}")
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
        query_inputs = self.tokenizer(
            query, return_tensors="np", padding=True, truncation=True, max_length=512
        )
        logger.debug(
            f"Query input IDs shape: {query_inputs['input_ids'].shape}, dtype: {query_inputs['input_ids'].dtype}")
        query_input_ids = mx.array(query_inputs["input_ids"]).astype(mx.int32)
        query_output = self.model(query_input_ids)
        logger.debug(
            f"Query output shape: {query_output.shape}, dtype: {query_output.dtype}")
        query_embedding = np.array(
            mx.mean(query_output, axis=1).tolist(), dtype=np.float32).squeeze()
        logger.debug(
            f"Query embedding shape: {query_embedding.shape}, dtype: {query_embedding.dtype}")
        norm_embeddings = embeddings / \
            np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_query = query_embedding / np.linalg.norm(query_embedding)
        logger.debug(
            f"Norm query shape: {norm_query.shape}, Norm embeddings shape: {norm_embeddings.shape}")
        similarities = np.dot(norm_embeddings, norm_query)
        logger.debug(f"Similarities shape: {similarities.shape}")
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        for idx in top_indices:
            similarity_score = similarities[idx]
            label: Literal["relevant",
                           "non-relevant"] = "relevant" if similarity_score >= relevance_threshold else "non-relevant"
            logger.debug(
                f"Streaming chunk index {idx}, score: {similarity_score:.4f}, label: {label}")
            yield label, similarity_score, idx  # Yield idx along with label and score
        logger.info("Streaming classification completed successfully")
