from jet.logger import logger
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoConfig
from span_marker import SpanMarkerModel
from span_marker.tokenizer import SpanMarkerTokenizer
from typing import Any, Dict, List, Union, Optional
import numpy as np
from tqdm import tqdm
import math


class EmbeddingGenerator:
    """A flexible class to generate embeddings using various model types with enhanced progress tracking."""

    def __init__(
        self,
        model_name: str,
        model_type: Optional[str] = None,
        use_mps: bool = True
    ):
        """
        Initialize the embedding generator with a specified model.

        Args:
            model_name: Name or path of the model (Hugging Face, SentenceTransformer, or SpanMarker)
            model_type: Optional model type ('sentence_transformer', 'causal', 'auto', 'span_marker')
            use_mps: Whether to use MPS (Apple Silicon GPU) if available
        """
        self.model_name = model_name
        self.device = 'mps' if use_mps and torch.backends.mps.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")

        # Auto-detect model type if not specified
        self.model_type = model_type or self._detect_model_type()

        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()
        self.model = self.model.to(self.device)
        self.model.eval()

        # Set precision for MPS
        self._original_dtype = torch.get_default_dtype()
        if self.device == 'mps':
            torch.set_default_dtype(torch.float16)

    def _detect_model_type(self) -> str:
        """Detect model type based on model name or configuration."""
        try:
            config = AutoConfig.from_pretrained(self.model_name)
            if 'sentence-transformers' in self.model_name:
                return 'sentence_transformer'
            elif 'span-marker' in self.model_name:
                return 'span_marker'
            elif config.model_type in ['gpt2', 'llama', 'mistral']:
                return 'causal'
            else:
                return 'auto'  # Generic transformer model
        except Exception as e:
            logger.warning(
                f"Could not detect model type: {str(e)}. Defaulting to sentence_transformer.")
            return 'sentence_transformer'

    def _load_model(self):
        """Load model and tokenizer based on model type."""
        try:
            if self.model_type == 'sentence_transformer':
                model = SentenceTransformer(self.model_name)
                tokenizer = None  # SentenceTransformer handles tokenization internally
            elif self.model_type == 'span_marker':
                model = SpanMarkerModel.from_pretrained(self.model_name)
                # Explicitly load config to ensure tokenizer has necessary attributes
                config = AutoConfig.from_pretrained(self.model_name)
                if not hasattr(config, 'model_max_length'):
                    config.model_max_length = 256  # Default value if not specified
                tokenizer = SpanMarkerTokenizer.from_pretrained(
                    self.model_name, config=config)
            else:
                model = AutoModel.from_pretrained(self.model_name)
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if self.model_type == 'causal' and tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            return model, tokenizer
        except Exception as e:
            logger.error(
                f"Error loading model or tokenizer for {self.model_name}: {str(e)}")
            raise

    def _batch_iterator(self, texts: List[str], batch_size: int) -> List[List[str]]:
        """Create batches from texts for processing."""
        return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    def _encode_causal(self, texts: List[str], max_length: int, batch_size: int, normalize: bool) -> np.ndarray:
        """Encode texts using a causal language model with progress tracking."""
        embeddings = []
        n_batches = math.ceil(len(texts) / batch_size)
        progress_bar = tqdm(self._batch_iterator(
            texts, batch_size), total=n_batches, desc="Encoding causal model")

        for batch in progress_bar:
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, -1, :].cpu(
                ).numpy()
                if normalize:
                    batch_embeddings = batch_embeddings / \
                        np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            embeddings.append(batch_embeddings)
            progress_bar.set_postfix(
                {"batch_size": len(batch), "embedding_dim": batch_embeddings.shape[1]})

        return np.concatenate(embeddings, axis=0)

    def _encode_transformer(self, texts: List[str], max_length: int, batch_size: int, normalize: bool) -> np.ndarray:
        """Encode texts using a generic transformer model (mean pooling) with progress tracking."""
        embeddings = []
        n_batches = math.ceil(len(texts) / batch_size)
        progress_bar = tqdm(self._batch_iterator(
            texts, batch_size), total=n_batches, desc="Encoding transformer model")

        for batch in progress_bar:
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(
                    -1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(
                    token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                if normalize:
                    batch_embeddings = batch_embeddings / \
                        np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            embeddings.append(batch_embeddings)
            progress_bar.set_postfix(
                {"batch_size": len(batch), "embedding_dim": batch_embeddings.shape[1]})

        return np.concatenate(embeddings, axis=0)

    def _encode_span_marker(self, texts: List[str], max_length: int, batch_size: int, normalize: bool) -> List[List[dict]]:
        """Process texts using SpanMarker model for NER with progress tracking."""
        results = []
        n_batches = math.ceil(len(texts) / batch_size)
        progress_bar = tqdm(self._batch_iterator(
            texts, batch_size), total=n_batches, desc="Encoding span marker model")

        for batch in progress_bar:
            try:
                # SpanMarkerModel handles tokenization internally
                predictions = self.model.predict(batch)
                batch_results = []
                for pred in predictions:
                    # Convert SpanMarker predictions to standardized format
                    batch_results.append([
                        {
                            "span": entity["span"],
                            "label": entity["label"],
                            "score": entity["score"],
                            "char_start_index": entity["char_start_index"],
                            "char_end_index": entity["char_end_index"]
                        } for entity in pred
                    ])
                results.extend(batch_results)
                progress_bar.set_postfix({"batch_size": len(
                    batch), "entities_found": sum(len(r) for r in batch_results)})
            except Exception as e:
                logger.error(
                    f"Error processing batch in span_marker: {str(e)}")
                raise
        return results

    def chunk_text(
        self,
        text: str,
        max_length: int = 256,
        stride: int = 128
    ) -> List[Dict[str, Any]]:
        """Chunk long text into overlapping segments based on tokenizer limits."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized for chunking.")
        try:
            tokens = self.tokenizer.encode_plus(
                text,
                return_offsets_mapping=True,
                return_tensors=None,
                add_special_tokens=True,
                max_length=None,
                return_attention_mask=False
            )
            input_ids = tokens["input_ids"]
            offset_mapping = tokens["offset_mapping"]
            chunks = []
            for i in range(0, len(input_ids), max_length - stride):
                start_idx = i
                end_idx = min(i + max_length, len(input_ids))
                chunk_input_ids = input_ids[start_idx:end_idx]
                chunk_offsets = offset_mapping[start_idx:end_idx]
                char_start = chunk_offsets[0][0] if chunk_offsets else 0
                char_end = chunk_offsets[-1][1] if chunk_offsets else len(text)
                chunk_text = self.tokenizer.decode(
                    chunk_input_ids, skip_special_tokens=True)
                chunks.append({
                    "text": chunk_text,
                    "input_ids": chunk_input_ids,
                    "offset_mapping": chunk_offsets,
                    "char_start": char_start,
                    "char_end": char_end
                })
            return chunks
        except Exception as e:
            logger.error(f"Failed to chunk text: {e}")
            raise

    def generate_embeddings(
        self,
        documents: List[str],
        batch_size: int = 3,
        max_length: int = 256,
        normalize: bool = True
    ) -> Union[np.ndarray, List[List[dict]]]:
        """
        Generate embeddings or entity predictions for a list of documents with detailed progress tracking.

        Args:
            documents: List of documents to encode
            batch_size: Batch size for encoding
            max_length: Maximum token length for the model
            normalize: Whether to normalize embeddings (ignored for span_marker)

        Returns:
            numpy array of shape (n_documents, embedding_dim) or list of entity predictions for span_marker
        """
        if not documents:
            raise ValueError("Document list cannot be empty")

        try:
            if self.model_type == 'sentence_transformer':
                embeddings = []
                batches = list(self._batch_iterator(documents, batch_size))
                progress_bar = tqdm(
                    batches,
                    total=len(batches),
                    desc="Encoding sentence transformer"
                )
                for batch in progress_bar:
                    with torch.no_grad():
                        batch_embeddings = self.model.encode(
                            batch,
                            batch_size=batch_size,
                            max_length=max_length,
                            normalize_embeddings=normalize,
                            convert_to_numpy=True,
                            show_progress_bar=False
                        )
                    embeddings.append(batch_embeddings)
                    progress_bar.set_postfix(
                        {"batch_size": len(batch), "embedding_dim": batch_embeddings.shape[1]})
                return np.concatenate(embeddings, axis=0)
            elif self.model_type == 'causal':
                return self._encode_causal(documents, max_length, batch_size, normalize)
            elif self.model_type == 'span_marker':
                return self._encode_span_marker(documents, max_length, batch_size, normalize)
            else:
                return self._encode_transformer(documents, max_length, batch_size, normalize)

        except Exception as e:
            logger.error(f"Error during embedding generation: {str(e)}")
            raise

        finally:
            # Clear memory
            if self.device == 'mps':
                torch.set_default_dtype(self._original_dtype)
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if self.device == 'mps':
                    torch.mps.empty_cache()
            except Exception as e:
                logger.warning(f"Error clearing memory: {str(e)}")
