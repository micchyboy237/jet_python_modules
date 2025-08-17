import logging
from typing import Optional, Union, Dict, List, ContextManager
from jet.models.model_types import MLXTokenizer
from contextlib import contextmanager
import mlx.core as mx

logger = logging.getLogger(__name__)


@contextmanager
def apply_logit_bias(
    logit_bias: Optional[Union[Dict[int, float], Dict[str, float], str, List[str]]],
    tokenizer: MLXTokenizer,
    default_bias: float = 15.0
) -> Dict[int, float]:
    """
    Context manager to convert and apply logit bias, ensuring it resets afterward.

    Args:
        logit_bias: Input logit bias as dictionary (int or str to float), string, or list of strings
        tokenizer: PreTrainedTokenizer instance for encoding strings
        default_bias: Default bias value for encoded tokens (default: 15.0)

    Yields:
        Dictionary mapping token IDs to bias values, or None if input is None
    """
    # Convert logit bias
    processed_logit_bias = convert_logit_bias(
        logit_bias, tokenizer, default_bias)

    # Yield the processed bias for use in generation
    try:
        yield processed_logit_bias
    finally:
        # Reset is implicit in MLX as logit_bias is not stored; log the reset for clarity
        if processed_logit_bias:
            logger.info("Logit bias reset to None after generation.")


def convert_logit_bias(
    logit_bias: Optional[Union[Dict[int, float], Dict[str, float], str, List[str]]],
    tokenizer: MLXTokenizer,
    default_bias: float = 10.0
) -> Optional[Dict[int, float]]:
    """
    Convert logit_bias from dictionary (int or str to float), string, or list of strings to a dictionary of token IDs with bias values.

    Args:
        logit_bias: Input logit bias as dictionary (int or str to float), string, or list of strings
        tokenizer: MLXTokenizer instance for encoding strings
        default_bias: Default bias value for encoded tokens (default: 15.0)

    Returns:
        Dictionary mapping token IDs to bias values, or None if input is None
    """
    if logit_bias is None:
        return None

    # Handle dictionary of int to float
    if isinstance(logit_bias, dict) and all(isinstance(k, int) for k in logit_bias.keys()):
        return logit_bias

    # Handle dictionary of str to float
    if isinstance(logit_bias, dict) and all(isinstance(k, str) for k in logit_bias.keys()):
        processed_logit_bias: Dict[int, float] = {}
        for token_str, bias in logit_bias.items():
            encoded = tokenizer.encode(token_str, add_special_tokens=False)
            if encoded:  # Check if encoding produced valid tokens
                processed_logit_bias[encoded[0]] = bias
            else:
                logger.warning(
                    f"Failed to encode logit_bias choice: {token_str}")
        return processed_logit_bias if processed_logit_bias else None

    # Convert single string to list for uniform processing
    if isinstance(logit_bias, str):
        logit_bias = [logit_bias]

    # Handle list of strings
    processed_logit_bias: Dict[int, float] = {}
    for choice in logit_bias:
        encoded = tokenizer.encode(choice, add_special_tokens=False)
        if encoded:  # Check if encoding produced valid tokens
            processed_logit_bias[encoded[0]] = default_bias
        else:
            logger.warning(f"Failed to encode logit_bias choice: {choice}")

    # Convert to MLX array for compatibility
    if processed_logit_bias:
        processed_logit_bias = {k: mx.array(
            v, dtype=mx.float32) for k, v in processed_logit_bias.items()}

    return processed_logit_bias if processed_logit_bias else None
