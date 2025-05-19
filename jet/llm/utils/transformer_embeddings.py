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
import tempfile
import pickle
from transformers import AutoTokenizer, AutoModel
import torch.utils.checkpoint as checkpoint


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


def calculate_dynamic_batch_size(embedding_dim: int, device: str, available_memory: float, model_key: str) -> int:
    """Calculate dynamic batch size based on embedding dimension, device, and model."""
    target_memory = available_memory * \
        0.6 if device in ["cuda", "mps"] else available_memory * 0.3
    bytes_per_embedding = embedding_dim * 2  # FP16 uses 2 bytes per element
    if "mxbai-embed-large" in model_key:
        target_memory *= 0.5
    batch_size = int(target_memory / (bytes_per_embedding * 1.5))
    return max(1, min(16 if "mxbai-embed-large" in model_key else 64, batch_size))


def chunk_texts(texts: Union[str, List[str]], max_tokens: int = 128) -> Tuple[List[str], List[int]]:
    """Chunk large texts and track original document indices."""
    if isinstance(texts, str):
        texts = [texts]
    chunked_texts = []
    doc_indices = []  # Tracks which document each chunk belongs to
    for doc_idx, text in enumerate(texts):
        words = text.split()
        if len(words) > max_tokens:
            for i in range(0, len(words), max_tokens):
                chunked_texts.append(" ".join(words[i:i + max_tokens]))
                doc_indices.append(doc_idx)
        else:
            chunked_texts.append(text)
            doc_indices.append(doc_idx)
    return chunked_texts, doc_indices


def generate_embeddings(
    model_key: EmbedModelType,
    texts: Union[str, List[str]],
    batch_size: Optional[int] = None,
    normalize: bool = True,
    _model: Optional[AutoModel] = None,
    _tokenizer: Optional[AutoTokenizer] = None,
    use_tqdm: Optional[bool] = None,
    max_tokens: Optional[int] = None,
    aggregate: bool = True
) -> Union[List[float], List[List[float]]]:
    """Generate embeddings with optimized memory usage for MPS and large models."""
    if not texts:
        raise ValueError("Input texts cannot be empty")

    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    device = "mps" if torch.backends.mps.is_available(
    ) else "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = resolve_model_key(model_key)
    model_id = AVAILABLE_EMBED_MODELS[embed_model]
    tokenizer = _tokenizer or AutoTokenizer.from_pretrained(model_id)
    model = _model or AutoModel.from_pretrained(
        model_id, torch_dtype=torch.float16).to(device)
    model.eval()

    embedding_dim = model.config.hidden_size
    is_single_text = isinstance(texts, str)
    if is_single_text:
        texts = [texts]

    if not max_tokens:
        max_tokens = get_embedding_size(embed_model)

    # Chunk texts and get document indices
    chunked_texts, doc_indices = chunk_texts(texts, max_tokens=max_tokens)
    num_original_texts = len(texts)

    if batch_size is not None and batch_size <= 0:
        raise ValueError("Batch size must be positive")

    batch_size = 64  # TODO: Update to calculate dynamically

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    temp_file_path = temp_file.name
    temp_file.close()

    try:
        if use_tqdm is None:
            use_tqdm = len(chunked_texts) > 2
        iterator = tqdm(range(0, len(chunked_texts), batch_size), desc="Generating embeddings",
                        leave=True) if use_tqdm else range(0, len(chunked_texts), batch_size)

        log_file = os.path.join(tempfile.gettempdir(), "mps_memory.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        with torch.autocast(device_type=device, dtype=torch.float16):
            for i in iterator:
                batch = chunked_texts[i:i + batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True,
                                   max_length=max_tokens, return_tensors="pt").to(device)

                with torch.no_grad():
                    def forward_pass(inputs):
                        return model(**inputs)
                    outputs = checkpoint.checkpoint(
                        forward_pass, inputs, use_reentrant=False)

                embeddings = outputs.last_hidden_state.mean(dim=1)
                if normalize:
                    embeddings = torch.nn.functional.normalize(
                        embeddings, p=2, dim=1)

                embeddings = embeddings.detach().cpu()

                with open(temp_file_path, "ab") as f:
                    pickle.dump(embeddings, f)

                del inputs, outputs, embeddings
                torch.mps.empty_cache()
                gc.collect()
            logger.debug(
                f"MPS memory allocated: {torch.mps.current_allocated_memory() / 1024**3:.2f} GB")

        logger.removeHandler(file_handler)
        file_handler.close()

        all_embeddings = []
        with open(temp_file_path, "rb") as f:
            while True:
                try:
                    embeddings = pickle.load(f)
                    all_embeddings.append(embeddings)
                except EOFError:
                    break

        all_embeddings = torch.cat(all_embeddings, dim=0)

        if aggregate:
            aggregated_embeddings = []
            for doc_idx in range(num_original_texts):
                chunk_mask = [i for i, idx in enumerate(
                    doc_indices) if idx == doc_idx]
                if chunk_mask:
                    chunk_embeddings = all_embeddings[chunk_mask]
                    doc_embedding = torch.mean(chunk_embeddings, dim=0)
                    aggregated_embeddings.append(doc_embedding)
                else:
                    logger.warning(
                        f"No chunks for document {doc_idx}, using zero embedding")
                    aggregated_embeddings.append(torch.zeros(embedding_dim))
            all_embeddings = torch.stack(aggregated_embeddings)
            result = all_embeddings.tolist()
        else:
            result = (all_embeddings.tolist(), doc_indices)

        del chunked_texts, doc_indices
        gc.collect()
        torch.mps.empty_cache()
        return result[0] if is_single_text and aggregate else result

    finally:
        try:
            os.unlink(temp_file_path)
        except OSError:
            pass


def get_embedding_function(
    model_name: str,
    batch_size: Optional[int] = None,
    normalize: bool = True,
    max_tokens: Optional[int] = None
) -> Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]]:
    """Load a Hugging Face model and tokenizer and return a callable that generates embeddings."""
    logger.info(f"Loading model: {model_name}")
    if model_name not in AVAILABLE_EMBED_MODELS:
        raise ValueError(
            f"Model {model_name} not found in AVAILABLE_EMBED_MODELS. Available models: {list(AVAILABLE_EMBED_MODELS.keys())}")
    model_id = AVAILABLE_EMBED_MODELS[model_name]
    device = "mps" if torch.backends.mps.is_available(
    ) else "cuda" if torch.cuda.is_available() else "cpu"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.float16).to(device)
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
            _tokenizer=tokenizer,
            max_tokens=max_tokens
        )
    return embedding_function


def search_docs(
    query: str,
    documents: List[str],
    model: EmbedModelType = "all-minilm:33m",
    top_k: int = 10,
    batch_size: Optional[int] = None,
    normalize: bool = True,
    max_tokens: Optional[int] = None
) -> List[SimilarityResult]:
    """Search documents with memory-efficient embedding generation and return SimilarityResult."""
    if not query or not documents:
        return []

    # Initialize tokenizer for token counting
    embed_model = resolve_model_key(model)
    model_id = AVAILABLE_EMBED_MODELS[embed_model]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    query_embedding = generate_embeddings(
        model, query, batch_size, normalize, max_tokens=max_tokens)
    doc_embeddings = generate_embeddings(
        model, documents, batch_size, normalize, max_tokens=max_tokens)

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

    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Convert indices to Python int to avoid NumPy integer types
    valid_indices = [int(idx) for idx in top_indices if idx < len(documents)]
    if not valid_indices:
        return []

    results = []
    for rank, idx in enumerate(valid_indices, start=1):
        doc_text = documents[idx]
        # Count tokens for the document
        tokens = len(tokenizer.encode(doc_text, add_special_tokens=True))
        result = SimilarityResult(
            id=f"doc_{idx}",
            rank=rank,
            doc_index=int(idx),  # Ensure Python int
            score=float(similarities[idx]),
            text=doc_text,
            tokens=tokens
        )
        results.append(result)

    del query_embedding, doc_embeddings, similarities
    torch.mps.empty_cache()
    gc.collect()
    return results


def cleanup():
    # Clear MPS memory
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        logger.debug(
            f"MPS memory cleared: {torch.mps.current_allocated_memory() / 1024**3:.2f} GB")

    # Run garbage collection
    gc.collect()

    logger.info("Transformer embed models unloaded and memory cleared.")


# Register cleanup function to run at exit
atexit.register(cleanup)
