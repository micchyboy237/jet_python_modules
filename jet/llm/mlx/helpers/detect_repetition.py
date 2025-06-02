from typing import List, Union, Dict
from jet.llm.mlx.mlx_types import CompletionResponse
from jet.logger import logger
import re


def detect_repetition(
    current_sequence: str,
    new_chunk: str,
    context_size: int = 50,
    repetition_threshold: float = 0.8,
    min_repetition_length: int = 10
) -> bool:
    """
    Detects if the new chunk causes repetition in the current sequence.

    Args:
        current_sequence: The accumulated sequence so far
        new_chunk: The new chunk to append
        context_size: Number of characters to consider for repetition check
        repetition_threshold: Similarity ratio to consider as repetition
        min_repetition_length: Minimum length of repeated pattern

    Returns:
        bool: True if repetition is detected, False otherwise
    """
    if not new_chunk or len(new_chunk) < min_repetition_length:
        return False

    # Combine current sequence with new chunk
    full_text = current_sequence + new_chunk

    # Consider only the last context_size characters
    if len(full_text) > context_size:
        check_text = full_text[-context_size:]
    else:
        check_text = full_text

    # Split into words for pattern matching
    words = check_text.split()
    if len(words) < 2:
        return False

    # Check for repeated patterns
    for length in range(min_repetition_length, len(check_text) // 2 + 1):
        pattern = check_text[-length:]
        pattern_escaped = re.escape(pattern)

        # Find all occurrences of the pattern
        matches = [m.start() for m in re.finditer(pattern_escaped, check_text)]

        # If pattern appears more than once in the context
        if len(matches) > 1:
            # Calculate similarity score based on pattern frequency
            pattern_length = len(pattern)
            total_length = len(check_text)
            similarity = (len(matches) * pattern_length) / total_length

            if similarity >= repetition_threshold:
                logger.warning(
                    f"Repetition detected: pattern '{pattern}' appears {len(matches)} times")
                return True

    return False
