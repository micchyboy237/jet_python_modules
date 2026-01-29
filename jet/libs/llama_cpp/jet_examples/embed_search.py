#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.logger import logger


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit length.
    Safe for zero vectors.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_texts(
    client: OpenAI,
    model: LLAMACPP_EMBED_KEYS,
    texts: Sequence[str],
    *,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Generic embedding helper.
    - Accepts any iterable of strings
    - Returns numpy array [N, D]
    - Processes in batches
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    all_vectors: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        response = client.embeddings.create(
            model=model,
            input=batch,
        )

        for item in response.data:
            all_vectors.append(item.embedding)

    return np.array(all_vectors, dtype=np.float32)


def cosine_similarity_matrix(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between two normalized matrices.
    Shape: [len(a), len(b)]
    """
    return a @ b.T


def main() -> None:
    # ------------------------------------------------------------
    # Sample inputs (variety of cases)
    # ------------------------------------------------------------
    query: str = "How do I build a fast vector search system?"

    documents: list[str] = [
        # ------------------------------------------------------------
        # Vector search / embeddings
        # ------------------------------------------------------------
        "Vector databases store high-dimensional embeddings for similarity search.",
        "Cosine similarity measures the angle between two normalized vectors.",
        "Dot product similarity is equivalent to cosine similarity for unit vectors.",
        "Embedding models transform text into dense numerical representations.",
        "Approximate nearest neighbor algorithms trade accuracy for speed.",
        "FAISS supports IVF, HNSW, and flat indexes for vector search.",
        "HNSW graphs provide fast recall with logarithmic search complexity.",
        "Dimensionality reduction can improve storage efficiency.",
        "Normalization is important when using cosine similarity.",
        "Batching embedding requests improves throughput.",
        # ------------------------------------------------------------
        # LLM / inference infrastructure
        # ------------------------------------------------------------
        "llama.cpp allows running large language models locally.",
        "Quantization reduces model size at the cost of some accuracy.",
        "GGUF is a file format optimized for llama.cpp inference.",
        "Context length determines how much text the model can process.",
        "Streaming responses reduce perceived latency.",
        "GPU acceleration can significantly improve inference speed.",
        "Memory bandwidth is often the bottleneck for inference.",
        "Model parallelism splits weights across devices.",
        "Prompt engineering affects output quality.",
        "Temperature controls output randomness.",
        # ------------------------------------------------------------
        # Software engineering
        # ------------------------------------------------------------
        "Modular code is easier to test and maintain.",
        "DRY stands for Don't Repeat Yourself.",
        "Unit tests validate small, isolated pieces of logic.",
        "Integration tests verify interactions between components.",
        "Static typing improves code readability.",
        "Logging helps diagnose production issues.",
        "Idempotent functions can be safely retried.",
        "Refactoring improves structure without changing behavior.",
        "Continuous integration automates testing pipelines.",
        "Version control tracks changes over time.",
        # ------------------------------------------------------------
        # Data / ML concepts
        # ------------------------------------------------------------
        "Overfitting occurs when a model memorizes training data.",
        "Regularization helps prevent overfitting.",
        "Training data quality affects model performance.",
        "Feature scaling improves optimization stability.",
        "Cross-validation estimates generalization error.",
        "Class imbalance can bias model predictions.",
        "Precision and recall measure classification quality.",
        "Loss functions guide optimization.",
        "Gradient descent minimizes loss iteratively.",
        "Inference differs from training.",
        # ------------------------------------------------------------
        # General knowledge (non-related distractors)
        # ------------------------------------------------------------
        "The Pacific Ocean is the largest ocean on Earth.",
        "Photosynthesis converts sunlight into chemical energy.",
        "Mount Everest is the tallest mountain above sea level.",
        "The human brain contains billions of neurons.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Lightning is a discharge of electricity.",
        "Saturn is known for its ring system.",
        "DNA carries genetic information.",
        "The speed of light is approximately 299,792 kilometers per second.",
        "Earth revolves around the Sun.",
        # ------------------------------------------------------------
        # Natural language variety
        # ------------------------------------------------------------
        "This sentence is intentionally vague and abstract.",
        "A very short sentence.",
        "An extremely long sentence " * 20,
        "Why does similarity search work so well in practice?",
        "🚀 Emojis and symbols should embed correctly.",
        "Numbers like 1234567890 also matter.",
        "Mixed-language text: English 日本語 Español.",
        "Newlines\nshould\nnot\nbreak\nembeddings.",
        "Tabs\tand\tspaces\tare\tfine.",
        "Punctuation!!! Should??? Not;;; Matter...",
        # ------------------------------------------------------------
        # Edge-like but valid inputs
        # ------------------------------------------------------------
        "A",
        " ",
        "     ",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    noisy_inputs: List[str] = [
        "",
        "   ",
        "Short.",
        "🚀 Unicode and emojis are fine.",
        "A" * 1000,  # long input stress test
    ]

    # ------------------------------------------------------------
    # Client setup (llama.cpp local server)
    # ------------------------------------------------------------
    client = OpenAI(
        base_url="http://shawn-pc.local:8081/v1",
        api_key="no-key-required",
        max_retries=3,
    )

    model: LLAMACPP_EMBED_KEYS = "nomic-embed-text"

    logger.info("Embedding model: %s", model)

    # ------------------------------------------------------------
    # Embed query
    # ------------------------------------------------------------
    logger.info("Embedding query...")
    query_vec = embed_texts(client, model, [query])
    query_vec = l2_normalize(query_vec)

    # ------------------------------------------------------------
    # Embed documents with progress bar
    # ------------------------------------------------------------
    logger.info("Embedding documents...")
    doc_vectors: List[np.ndarray] = []

    for text in tqdm(documents, desc="Embedding docs"):
        vec = embed_texts(client, model, [text])
        doc_vectors.append(vec[0])

    doc_matrix = l2_normalize(np.vstack(doc_vectors))

    # ------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------
    similarities = cosine_similarity_matrix(query_vec, doc_matrix)[0]

    ranked = sorted(
        zip(documents, similarities),
        key=lambda x: x[1],
        reverse=True,
    )

    logger.info("Top matches:")
    for rank, (text, score) in enumerate(ranked, start=1):
        logger.info("[%d] score=%.4f | %s", rank, score, text)

    # ------------------------------------------------------------
    # Edge / stress examples
    # ------------------------------------------------------------
    logger.info("Embedding noisy / edge-case inputs...")
    noisy_vectors = embed_texts(client, model, noisy_inputs)

    logger.info(
        "Noisy embeddings shape: %s",
        noisy_vectors.shape,
    )


if __name__ == "__main__":
    main()
