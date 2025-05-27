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
from transformers import AutoTokenizer, AutoModel
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


def chunk_texts(texts: Union[str, List[str]], chunk_size: int = 128) -> Tuple[List[str], List[int]]:
    """Chunk large texts and track original document indices."""
    if isinstance(texts, str):
        texts = [texts]
    chunked_texts = []
    doc_indices = []  # Tracks which document each chunk belongs to
    for doc_idx, text in enumerate(texts):
        words = text.split()
        if len(words) > chunk_size:
            for i in range(0, len(words), chunk_size):
                chunked_texts.append(" ".join(words[i:i + chunk_size]))
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
    chunk_size: Optional[int] = None,
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

    if not chunk_size:
        chunk_size = get_embedding_size(embed_model)

    # Chunk texts and get document indices
    chunked_texts, doc_indices = chunk_texts(texts, chunk_size=chunk_size)
    num_original_texts = len(texts)

    if batch_size is None:
        batch_size = 64  # Default batch size, suitable for Mac M1 MPS
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    # Use DataLoader for batching
    class TextDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx]

    dataset = TextDataset(chunked_texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if use_tqdm is None:
        use_tqdm = len(chunked_texts) > 2

    all_embeddings = []
    logger.debug(
        f"Processing {len(chunked_texts)} texts with batch_size={batch_size} on {device}")

    with torch.autocast(device_type=device, dtype=torch.float16):
        for batch in tqdm(dataloader, desc="Generating embeddings", leave=True) if use_tqdm else dataloader:
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=chunk_size, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                if normalize:
                    embeddings = torch.nn.functional.normalize(
                        embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu())
            del inputs, outputs, embeddings
            torch.mps.empty_cache()  # Clear MPS memory after each batch
            gc.collect()

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
    torch.mps.empty_cache()
    gc.collect()
    return result[0] if is_single_text and aggregate else result


def get_embedding_function(
    model_name: str,
    batch_size: Optional[int] = None,
    normalize: bool = True,
    chunk_size: Optional[int] = None
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
            chunk_size=chunk_size
        )
    return embedding_function


def search_docs(
    query: str,
    documents: List[str],
    model: EmbedModelType = "all-minilm:33m",
    top_k: Optional[int] = 10,
    batch_size: Optional[int] = None,
    normalize: bool = True,
    chunk_size: Optional[int] = None,
    ids: Optional[List[str]] = None
) -> List[SimilarityResult]:
    """Search documents with memory-efficient embedding generation and return SimilarityResult."""
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
        # Use provided ID if available, otherwise default to f"doc_{idx}"
        doc_id = ids[idx] if ids is not None else f"doc_{idx}"
        result = SimilarityResult(
            id=doc_id,
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
