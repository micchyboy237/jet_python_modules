from typing import List, Dict, Any, Literal, TypedDict, Union
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, SimilarityFunction, models
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig
from jet.file.utils import save_file
from jet.models.model_types import EmbedModelType, ModelType
from jet.models.config import HF_TOKEN, MODELS_CACHE_DIR

from jet.logger import logger
from jet.models.tokenizer.base import count_tokens, get_max_token_count
from jet.models.utils import resolve_model_value


EMBED_MODEL: EmbedModelType = "all-MiniLM-L6-v2"


class SimilarityResult(TypedDict):
    """
    Represents a single similarity result for a text.

    Fields:
        id: Identifier for the text.
        rank: Rank based on score (1 for highest, no skips).
        doc_index: Original index of the text in the input list.
        score: Normalized similarity score.
        text: The compared text (or chunk if long).
        tokens: Number of tokens from text.
        matched: Dictionary mapping matched query terms to their counts.
    """
    id: str
    rank: int
    doc_index: int
    score: float
    text: str
    tokens: int
    matched: Dict[str, int]


def compute_similarity_results(
    model: SentenceTransformer,
    query: str,
    passages: List[str],
    similarity_fn: SimilarityFunction = "cosine"
) -> List[SimilarityResult]:
    """
    Compute similarity results between a query and passages.

    Args:
        model: SentenceTransformer model.
        query: Query text.
        passages: List of passage texts.
        similarity_fn: Similarity function name (e.g., "cosine").

    Returns:
        List of SimilarityResult dictionaries.
    """
    # Encode query and passages
    query_embedding = model.encode([query], convert_to_tensor=True)
    passage_embeddings = model.encode(passages, convert_to_tensor=True)

    # Compute similarities
    similarities = model.similarity(query_embedding, passage_embeddings)[0]
    similarities_np = similarities.cpu().numpy()

    # Get token counts
    tokenizer = model.tokenizer
    passage_tokens = [
        len(tokenizer.encode(passage, add_special_tokens=True)) for passage in passages
    ]

    # Compute ranks (1 for highest, no skips)
    sorted_indices = np.argsort(-similarities_np)  # Descending order
    ranks = np.zeros(len(passages), dtype=int)
    for rank, idx in enumerate(sorted_indices, 1):
        ranks[idx] = rank

    # Basic matched terms (split query into words, count occurrences)
    query_terms = query.lower().split()
    results = []
    for idx, (passage, score, rank, tokens) in enumerate(
        zip(passages, similarities_np, ranks, passage_tokens)
    ):
        matched = {
            term: passage.lower().count(term) for term in query_terms if term in passage.lower()
        }
        result: SimilarityResult = {
            "id": f"doc_{idx}",
            "rank": rank,
            "doc_index": idx,
            "score": float(score),
            "text": passage,
            "tokens": tokens,
            "matched": matched,
        }
        results.append(result)

    return results


def load_pretrained_model_with_default_settings() -> List[SimilarityResult]:
    """
    Demonstrates loading a pre-trained SentenceTransformer model with default settings.
    Uses: model_name_or_path, device
    Scenario: Basic text embedding for general-purpose sentence similarity.
    """
    # Initialize model with a pre-trained model name and device
    model = SentenceTransformer(
        model_name_or_path=EMBED_MODEL,
        backend="onnx",
        device="cpu",
    )

    # Define query and sentences
    query = "I enjoy coding."
    sentences = ["I enjoy coding.",
                 "Programming is fun!", "I like to read books."]

    # Compute similarity results
    results = compute_similarity_results(model, query, sentences)
    logger.gray("\nPre-trained model similarities:")
    logger.success(results)

    return results


def create_custom_model_with_modules() -> List[SimilarityResult]:
    """
    Build a custom SentenceTransformer using standard modules.
    TokenizerModule removed — handled separately as SentenceTransformer expects.
    """
    # Define query and sentences
    query = "This is a test."
    sentences = ["This is a test.", "Another test sentence."]

    # Compute dynamic max_seq_length
    max_seq_length = get_max_token_count("bert-base-uncased", sentences)

    # Step 1: Tokenizer and transformer
    word_embedding_model = models.Transformer(
        "bert-base-uncased", max_seq_length=max_seq_length)

    # Step 2: Pooling layer
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )

    # Step 3: Combine into model
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Compute similarity results
    results = compute_similarity_results(model, query, sentences)
    logger.gray("\nCustom model similarities:")
    logger.success(results)

    return results


def use_prompts_for_contextual_encoding() -> List[SimilarityResult]:
    """
    Demonstrates using prompts and default_prompt_name for contextual encoding.
    Uses: model_name_or_path, prompts, default_prompt_name, device
    Scenario: Encoding queries and passages for a search engine with role-specific prompts.
    """
    prompts = {
        "query": "query: ",
        "passage": "passage: "
    }
    model = SentenceTransformer(
        model_name_or_path="multi-qa-MiniLM-L6-cos-v1",
        prompts=prompts,
        default_prompt_name="query",
        backend="onnx",
        device="cpu"
    )
    query = "What is Python?"
    passages = [
        "Python is a programming language.",
        "Java is used for enterprise applications.",
        "Python is great for data science."
    ]
    results = compute_similarity_results(model, query, passages)
    logger.gray("\nPre-trained model similarities:")
    logger.success(results)

    return results


def load_private_model_with_auth() -> List[SimilarityResult]:
    """
    Demonstrates loading a private model from Hugging Face with authentication.
    Uses: model_name_or_path, token, device, cache_folder
    Scenario: Accessing a private model for a company-specific application.
    """
    # Initialize model with authentication
    model = SentenceTransformer(
        model_name_or_path=EMBED_MODEL,
        token=HF_TOKEN,
        cache_folder=MODELS_CACHE_DIR,
        backend="onnx",
        device="cpu",
    )

    # Define query and sentences
    query = "Confidential data analysis."
    sentences = ["Confidential data analysis.", "Secure text processing."]

    # Compute similarity results
    results = compute_similarity_results(model, query, sentences)
    logger.gray("\nPrivate model similarities:")
    logger.success(results)

    return results


def use_optimized_backend_with_truncation() -> List[SimilarityResult]:
    """
    Demonstrates using an optimized backend (ONNX) with truncated embeddings.
    Uses: model_name_or_path, backend, truncate_dim, model_kwargs, device
    Scenario: Deploying an efficient model for low-latency inference.
    """
    # Define query and sentences
    query = "Fast inference test."
    sentences = ["Fast inference test.", "Optimized model performance."]

    # Compute dynamic truncate_dim
    truncate_dim = get_max_token_count(EMBED_MODEL, sentences)

    # Define model kwargs for ONNX
    model_kwargs = {
        "provider": "CPUExecutionProvider",
        "export": True
    }

    # Initialize model with ONNX backend
    model = SentenceTransformer(
        model_name_or_path=EMBED_MODEL,
        backend="onnx",
        truncate_dim=truncate_dim,
        model_kwargs=model_kwargs,
        device="cpu"
    )

    # Encode sentences to verify truncation
    embeddings = model.encode(sentences)
    print(f"Truncated embedding shape: {embeddings.shape}")

    # Compute similarity results
    results = compute_similarity_results(model, query, sentences)
    logger.gray("\nONNX backend similarities:")
    logger.success(results)

    return results


def load_specific_revision_with_custom_config() -> List[SimilarityResult]:
    """
    Demonstrates loading a specific model revision with custom configuration.
    Uses: model_name_or_path, revision, config_kwargs, tokenizer_kwargs, trust_remote_code
    Scenario: Using a specific model version for reproducibility in research.
    """
    # Define custom config and tokenizer kwargs
    config_kwargs = {"hidden_dropout_prob": 0.2}
    tokenizer_kwargs = {"use_fast": True}

    # Initialize model with specific revision
    model = SentenceTransformer(
        model_name_or_path=EMBED_MODEL,
        revision="main",
        config_kwargs=config_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
        trust_remote_code=True,
        backend="onnx",
        device="cpu"
    )

    # Define query and sentences
    query = "Reproducible research."
    sentences = ["Reproducible research.", "Consistent model version."]

    # Compute similarity results
    results = compute_similarity_results(model, query, sentences)
    logger.gray("\nSpecific revision similarities:")
    logger.success(results)

    return results


def main() -> None:
    """Run all example functions."""
    import os
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    print("Running SentenceTransformer Examples\n")
    results = load_pretrained_model_with_default_settings()
    save_file(
        results, f"{output_dir}/load_pretrained_model_with_default_settings.json")
    print("\n")
    results = create_custom_model_with_modules()
    save_file(results, f"{output_dir}/create_custom_model_with_modules.json")
    print("\n")
    results = use_prompts_for_contextual_encoding()
    save_file(
        results, f"{output_dir}/use_prompts_for_contextual_encoding.json")
    print("\n")
    results = load_private_model_with_auth()
    save_file(results, f"{output_dir}/load_private_model_with_auth.json")
    print("\n")
    results = use_optimized_backend_with_truncation()
    save_file(
        results, f"{output_dir}/use_optimized_backend_with_truncation.json")
    print("\n")
    results = load_specific_revision_with_custom_config()
    save_file(
        results, f"{output_dir}/load_specific_revision_with_custom_config.json")


if __name__ == "__main__":
    main()
