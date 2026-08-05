# jet_python_modules/jet/adapters/llama_cpp/chunk_strategies/model_utils.py
"""Model-aware utilities for chunking."""

import logging

from jet.adapters.llama_cpp.chunking_utils import _get_size_fn
from jet.adapters.llama_cpp.model_utils import get_model_ctx_embd_size
from jet.adapters.llama_cpp.models import LLAMACPP_MODEL_CONTEXTS
from jet.adapters.llama_cpp.types import LLAMACPP_KEYS

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_SIZE = 512
_DEFAULT_CTX_PERCENTAGE = 0.08  # ~328 tokens for 4096 ctx; lands in 256-400 sweet spot


def get_optimal_chunk_size(
    model: str | LLAMACPP_KEYS,
    ctx_percentage: float = _DEFAULT_CTX_PERCENTAGE,
    min_size: int = 128,
    max_size: int = 1024,
) -> int:
    """Get optimal chunk size from model context window.

    Resolution order:
    1. Static LLAMACPP_MODEL_CONTEXTS dict (instant, no network)
    2. Live server query via get_model_ctx_embd_size (fallback)
    3. Hardcoded default if both fail

    Default ctx_percentage=0.08 targets 256-400 tokens for typical small-context
    models (4096 ctx → 328 tokens), matching mid-2026 RAG best practices.

    Args:
        model: Model identifier, short alias, or LLAMACPP_KEYS enum.
        ctx_percentage: Fraction of context window to use per chunk.
        min_size: Minimum chunk size floor.
        max_size: Maximum chunk size ceiling.

    Returns:
        Optimal chunk size in tokens, clamped to [min_size, max_size].
    """
    if not 0.0 < ctx_percentage <= 1.0:
        logger.warning(
            "ctx_percentage %.2f out of range (0,1]; clamping to %.2f",
            ctx_percentage,
            _DEFAULT_CTX_PERCENTAGE,
        )
        ctx_percentage = _DEFAULT_CTX_PERCENTAGE

    ctx = 0

    if model in LLAMACPP_MODEL_CONTEXTS:
        ctx = LLAMACPP_MODEL_CONTEXTS[model]
        logger.debug("Static context for %s: %d tokens", model, ctx)
    else:
        try:
            info = get_model_ctx_embd_size(model)
            ctx = info.get("ctx", 0)
            logger.debug("Server context for %s: %d tokens", model, ctx)
        except Exception as exc:
            logger.warning(
                "Failed to get context size for model %s (%s: %s); falling back to %d",
                model,
                type(exc).__name__,
                exc,
                _DEFAULT_CHUNK_SIZE,
            )

    if ctx > 0:
        computed = int(ctx * ctx_percentage)
        result = min(max(computed, min_size), max_size)
        logger.debug(
            "Optimal chunk size for %s: ctx=%d, %.0f%% → %d (clamped [%d,%d])",
            model,
            ctx,
            ctx_percentage * 100,
            result,
            min_size,
            max_size,
        )
        return result

    return _DEFAULT_CHUNK_SIZE


def estimate_tokens_safe(
    text: str,
    model: str | LLAMACPP_KEYS | None = None,
    fallback_chars_per_token: int = 4,
) -> int:
    """Count tokens with graceful fallback to character-based estimation.

    Uses the same tokenizer backend as the chunking strategies for consistency.
    """
    if not text:
        return 0
    if model is None:
        return max(1, len(text) // fallback_chars_per_token)
    try:
        size_fn = _get_size_fn(model)
        tokens = size_fn(text)
        count = len(tokens) if isinstance(tokens, list) else tokens
        logger.debug(
            "Token count for %d chars (%s): %d tokens", len(text), model, count
        )
        return max(1, count)
    except Exception as exc:
        estimate = max(1, len(text) // fallback_chars_per_token)
        logger.warning(
            "Tokenizer failed for model %s (%s: %s); char-fallback estimate=%d",
            model,
            type(exc).__name__,
            exc,
            estimate,
        )
        return estimate
