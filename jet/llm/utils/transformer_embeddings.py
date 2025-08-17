import psutil
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import atexit
from jet.models.model_types import EmbedModelType
from jet.llm.mlx.models import AVAILABLE_EMBED_MODELS, get_context_size, resolve_model_key
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
    model: Optional[AutoModel] = None,
    tokenizer: Optional[AutoTokenizer] = None,
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
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_id)
    model = model or AutoModel.from_pretrained(
        model_id, torch_dtype=torch.float16).to(device)
    model.eval()

    embedding_dim = model.config.hidden_size
    is_single_text = isinstance(texts, str)
    if is_single_text:
        texts = [texts]

    if not chunk_size:
        chunk_size = get_context_size(embed_model)

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

    return result[0] if is_single_text and aggregate else result


def get_embedding_function(
    model_name: EmbedModelType,
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
            model=model,
            tokenizer=tokenizer,
            chunk_size=chunk_size
        )
    return embedding_function


def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.debug(f"Process memory usage: {mem_info.rss / 1024**3:.2f} GB")


def cleanup():
    # Clear MPS memory
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        logger.debug(
            f"MPS memory cleared: {torch.mps.current_allocated_memory() / 1024**3:.2f} GB")

    # Run garbage collection
    gc.collect()

    logger.info("Transformer embed models unloaded and memory cleared.")

    log_memory_usage()


# Register cleanup function to run at exit
atexit.register(cleanup)
