"""
Shared mock utilities for text_chunker demos.
Provides mocked tokenize, detokenize, and get_model_ctx_embd_size
so demos run without a live llama.cpp server.
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

mock_tokenize = MagicMock()
mock_detokenize = MagicMock()
mock_get_model_ctx_embd_size = MagicMock()

# Simple word ↔ ID mappings for realistic decode output
_word_to_id = {}
_id_to_word = {}
_counter = 0


def _get_word_id(word: str) -> int:
    global _counter
    if word not in _word_to_id:
        _counter += 1
        _word_to_id[word] = _counter
        _id_to_word[_counter] = word
    return _word_to_id[word]


def _setup_mocks():
    """Configure mock responses (idempotent)."""
    if mock_tokenize.side_effect is not None:
        return

    def tokenize_side_effect(content, model=None, add_special=False, **kwargs):
        # Handle both single string and list of strings
        if isinstance(content, list):
            # Batch: return list of token lists
            return [
                tokenize_side_effect(c, model, add_special)["tokens"] for c in content
            ]
        words = content.split()
        tokens = [{"id": _get_word_id(word), "piece": word} for word in words]
        return {"tokens": tokens}

    def detokenize_side_effect(tokens, model=None, **kwargs):
        if not tokens:
            return {"content": ""}
        if isinstance(tokens[0], dict):
            pieces = [t.get("piece", "") for t in tokens]
        else:
            pieces = [_id_to_word.get(t, f"w{t}") for t in tokens]
        return {"content": " ".join(pieces)}

    mock_tokenize.side_effect = tokenize_side_effect
    mock_detokenize.side_effect = detokenize_side_effect
    mock_get_model_ctx_embd_size.return_value = {
        "ctx": 2048,
        "ctx_train": 2048,
        "embd_dims": 768,
    }


def apply_mocks():
    """Apply mocks and return the text_chunker module."""
    _setup_mocks()
    patch_path = "jet.wordnet.text_chunker"
    patch(f"{patch_path}.tokenize", mock_tokenize).start()
    patch(f"{patch_path}.detokenize", mock_detokenize).start()
    patch(f"{patch_path}.get_model_ctx_embd_size", mock_get_model_ctx_embd_size).start()

    import jet.wordnet.text_chunker as tc

    return tc


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
