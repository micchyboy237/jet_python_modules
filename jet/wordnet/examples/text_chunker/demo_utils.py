"""
Shared mock utilities for text_chunker demos.
Provides mocked tokenize, detokenize, and get_model_ctx_embd_size
so demos run without a live llama.cpp server.
"""

import os
import sys
from unittest.mock import MagicMock, patch

# Ensure jet package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

mock_tokenize = MagicMock()
mock_detokenize = MagicMock()
mock_get_model_ctx_embd_size = MagicMock()


def _setup_mocks():
    """Configure mock responses for tokenize and detokenize (called once)."""
    if mock_tokenize.side_effect is not None:
        return  # Already set up

    def tokenize_side_effect(content, model=None, add_special=False, **kwargs):
        words = content.split()
        tokens = [{"id": (i % 100) + 1, "piece": word} for i, word in enumerate(words)]
        return {"tokens": tokens}

    def detokenize_side_effect(tokens, model=None, **kwargs):
        if isinstance(tokens[0], dict):
            pieces = [t.get("piece", "") for t in tokens]
        else:
            pieces = [f"w{t}" for t in tokens]
        return {"content": " ".join(pieces)}

    mock_tokenize.side_effect = tokenize_side_effect
    mock_detokenize.side_effect = detokenize_side_effect
    mock_get_model_ctx_embd_size.return_value = {
        "ctx": 2048,
        "ctx_train": 2048,
        "embd_dims": 768,
    }


def apply_mocks():
    """Apply mocks and return the text_chunker module ready to use."""
    _setup_mocks()
    patch_path = "jet.wordnet.text_chunker"
    patcher1 = patch(f"{patch_path}.tokenize", mock_tokenize)
    patcher2 = patch(f"{patch_path}.detokenize", mock_detokenize)
    patcher3 = patch(
        f"{patch_path}.get_model_ctx_embd_size", mock_get_model_ctx_embd_size
    )
    patcher1.start()
    patcher2.start()
    patcher3.start()

    # Import after patches are active
    import jet.wordnet.text_chunker as tc

    return tc


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
