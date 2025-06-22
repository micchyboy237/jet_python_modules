from tqdm import tqdm
from typing import List, Optional, Iterator, Literal
from jet.logger import logger
from jet.models.model_types import ModelType
from jet.models.utils import resolve_model_value
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from transformers import AutoTokenizer
import trafilatura
import spacy
from spacy.language import Language
import textacy.preprocessing as tprep
import numpy as np


class MLXRAGProcessor:
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
            inputs = self.tokenizer(
                batch_chunks,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=512
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


class WebDataPreprocessor:
    """Preprocesses web-scraped data for RAG usage."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """Initialize with chunking parameters and load Spacy model."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=[
                                  "ner", "lemmatizer"])
        except OSError:
            logger.error(
                "Spacy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            raise

    def fetch_and_extract(self, url: str) -> Optional[str]:
        """Fetch webpage and extract main content using trafilatura."""
        logger.info(f"Fetching and extracting content from {url}")
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                logger.warning(f"Failed to fetch content from {url}")
                return None
            extracted = trafilatura.extract(
                downloaded, include_comments=False, include_tables=False)
            if not extracted:
                logger.warning(f"No content extracted from {url}")
                return None
            logger.info(f"Successfully extracted content from {url}")
            return extracted
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return None

    def clean_text(self, text: str) -> str:
        """Clean extracted text for RAG processing using textacy."""
        logger.info("Cleaning text")
        cleaned = tprep.normalize.unicode(text, form="NFKC")
        cleaned = tprep.normalize.whitespace(cleaned)
        cleaned = tprep.replace.urls(cleaned, "")
        cleaned = tprep.replace.emails(cleaned, "")
        cleaned = tprep.replace.phone_numbers(cleaned, "")
        cleaned = tprep.remove.punctuation(cleaned)
        cleaned = " ".join(cleaned.split())
        logger.info("Text cleaning completed")
        return cleaned

    def chunk_text(self, text: str) -> List[str]:
        """Chunk text into semantically coherent pieces using Spacy."""
        logger.info("Chunking text")
        doc: Language = self.nlp(text)
        sentences = [sent.text.strip()
                     for sent in doc.sents if len(sent.text.strip()) > 10]
        if not sentences:
            logger.warning("No valid sentences found for chunking")
            return []
        text_to_chunk = " ".join(sentences)
        chunks = []
        start_idx = 0
        text_length = len(text_to_chunk)
        while start_idx < text_length:
            end_idx = min(start_idx + self.chunk_size, text_length)
            if end_idx < text_length:
                sub_doc = self.nlp(text_to_chunk[start_idx:end_idx])
                last_sent = list(sub_doc.sents)[-1] if sub_doc.sents else None
                if last_sent:
                    sentence_end = start_idx + last_sent.end_char
                    end_idx = min(sentence_end, end_idx)
            chunk = text_to_chunk[start_idx:end_idx].strip()
            if len(chunk) >= 10:
                chunks.append(chunk)
            start_idx = max(start_idx + self.chunk_size -
                            self.chunk_overlap, end_idx)
        logger.info(f"Generated {len(chunks)} valid chunks")
        return chunks

    def preprocess(self, url: str) -> List[str]:
        """Full preprocessing pipeline: fetch, clean, and chunk."""
        logger.info(f"Starting preprocessing for {url}")
        raw_text = self.fetch_and_extract(url)
        if not raw_text:
            logger.error(f"Preprocessing failed: No content from {url}")
            return []
        cleaned_text = self.clean_text(raw_text)
        if not cleaned_text:
            logger.warning(f"Preprocessing resulted in empty text for {url}")
            return []
        chunks = self.chunk_text(cleaned_text)
        logger.info(
            f"Preprocessing completed for {url} with {len(chunks)} chunks")
        return chunks
