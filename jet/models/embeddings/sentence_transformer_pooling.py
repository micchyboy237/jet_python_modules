# sentence_transformer_pooling.py

import logging
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer
from typing import Any, Dict, List, Literal, Tuple

from jet.logger import logger


PoolingMode = Literal["cls_token", "mean_tokens",
                      "max_tokens", "mean_sqrt_len_tokens"]
AttentionMode = Literal["eager", "sdpa"]

# Global model cache
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}


def _get_cache_key(model_name: str, pooling_mode: PoolingMode, attn_implementation: AttentionMode) -> str:
    """Generate a unique cache key for model configuration."""
    return f"{model_name}_{pooling_mode}_{attn_implementation}"


def load_sentence_transformer(
    model_name: str,
    pooling_mode: PoolingMode = "mean_tokens",
    attn_implementation: AttentionMode = "eager",
    model_kwargs: Dict[str, Any] | None = None,
) -> SentenceTransformer:
    """Load a SentenceTransformer with specified pooling mode."""
    cache_key = _get_cache_key(model_name, pooling_mode, attn_implementation)

    # Check if model is already in cache
    if cache_key in _MODEL_CACHE:
        logger.debug(f"Using cached model for key: {cache_key}")
        return _MODEL_CACHE[cache_key]

    try:
        logger.debug(
            f"Loading model {model_name} with pooling mode {pooling_mode}")

        # Choose device
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        logger.debug(f"Using device: {device}")

        # Correctly use the sentence_transformers Transformer wrapper
        transformer = Transformer(model_name).to(device)

        pooling_layer = Pooling(
            word_embedding_dimension=transformer.get_word_embedding_dimension(),
            pooling_mode_cls_token=(pooling_mode == "cls_token"),
            pooling_mode_mean_tokens=(pooling_mode == "mean_tokens"),
            pooling_mode_max_tokens=(pooling_mode == "max_tokens"),
            pooling_mode_mean_sqrt_len_tokens=(
                pooling_mode == "mean_sqrt_len_tokens")
        )

        model_kwargs = {
            "torch_dtype": torch.float16,
            "attn_implementation": attn_implementation,
            **(model_kwargs or {})
        }

        model = SentenceTransformer(
            modules=[transformer, pooling_layer], device=device, model_kwargs=model_kwargs)

        # Cache the model
        _MODEL_CACHE[cache_key] = model

        logger.info(
            f"Successfully loaded model {model_name} with {pooling_mode} pooling on {device}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise


def encode_sentences(model: SentenceTransformer, sentences: List[str], batch_size: int = 32) -> np.ndarray:
    """Encode sentences into embeddings."""
    try:
        logger.debug(
            f"Encoding {len(sentences)} sentences with batch size {batch_size}")
        embeddings = model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        logger.info(
            f"Encoded {len(sentences)} sentences into {embeddings.shape} embeddings")
        return embeddings
    except Exception as e:
        logger.error(f"Encoding failed: {str(e)}")
        raise


def search_docs(
    model: SentenceTransformer,
    corpus: List[str],
    query: str,
    top_k: int = 3
) -> List[Tuple[int, float, str]]:
    """Search corpus using cosine similarity and return top_k results with scores."""
    try:
        logger.debug(f"Searching top {top_k} results for query: {query}")
        corpus_embeddings = encode_sentences(model, corpus)
        query_embedding = encode_sentences(model, [query])[0]

        cosine_scores = np.dot(corpus_embeddings, query_embedding) / (
            np.linalg.norm(corpus_embeddings, axis=1) *
            np.linalg.norm(query_embedding) + 1e-8
        )

        top_indices = np.argsort(cosine_scores)[::-1][:top_k]
        results = [(i, float(cosine_scores[i]), corpus[i])
                   for i in top_indices]
        return results
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise


if __name__ == "__main__":
    sample_corpus = [
        "Our AI-powered analytics platform helps you understand customer behavior at scale, offering dashboards, anomaly detection, and churn prediction.",
        "This device complies with part 15 of the FCC Rules. Operation is subject to the following two conditions...",
        "A long time ago in a galaxy far, far away, a young farm boy discovers his destiny amidst intergalactic war and rebellion.",
        "At our law firm, we specialize in cross-border intellectual property litigation, patent prosecution, and trademark enforcement.",
        "We help ecommerce brands scale by building omnichannel marketing funnels using automation, audience segmentation, and real-time analytics.",
        "The software supports event-driven microservice architecture, written in TypeScript, deployable on AWS Lambda with DynamoDB and SQS integration."
    ]

    user_query = "How can I use data analytics to improve my ecommerce marketing strategy?"

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    pooling_strategies: List[PoolingMode] = [
        "cls_token",
        "mean_tokens",
        "max_tokens",
        "mean_sqrt_len_tokens"
    ]

    for strategy in pooling_strategies:
        print("=" * 80)
        model = load_sentence_transformer(model_name, pooling_mode=strategy)
        results = search_docs(model, sample_corpus, user_query)

        print(f"🔍 Pooling strategy: {strategy}")
        print(f"📌 Top results for query:\n'{user_query}'\n")
        for rank, (idx, score, text) in enumerate(results, 1):
            print(f"[{rank}] Score: {score:.4f}")
            print(f"{text}\n")
