"""
Custom Chonkie tokenizer backed by jet's llama.cpp token_utils.
Inherits from chonkie.tokenizer.Tokenizer so AutoTokenizer recognizes it.
"""

from typing import Sequence

from jet.adapters.llama_cpp.token_utils import (
    count_tokens,
    detokenize,
    tokenize,
)
from jet.adapters.llama_cpp.token_utils.tokenizer_management import get_tokenizer

from chonkie.tokenizer import Tokenizer


class LlamaCppTokenizer(Tokenizer):
    """
    Chonkie-compatible tokenizer using jet's llama.cpp token_utils.

    Inherits from chonkie.tokenizer.Tokenizer so that AutoTokenizer
    automatically wraps it as ChonkieAutoTokenizer.

    Usage:
        from jet.adapters.chonkie.llama_cpp_tokenizer import LlamaCppTokenizer
        from chonkie import RecursiveChunker

        tok = LlamaCppTokenizer(model="qwen2.5-7b-instruct")
        chunker = RecursiveChunker(tokenizer=tok, chunk_size=512)
        chunks = chunker.chunk(text)
    """

    def __init__(
        self,
        model: str | None = None,
        add_special_tokens: bool = False,
        skip_special_tokens: bool = True,
    ):
        super().__init__()  # Required: initializes vocab + token2id
        self.model = model
        self.add_special_tokens = add_special_tokens
        self.skip_special_tokens = skip_special_tokens
        # Eagerly load to fail fast if model is invalid
        self._tokenizer = get_tokenizer(model)

    def __repr__(self) -> str:
        return f"LlamaCppTokenizer(model={self.model!r})"

    def encode(self, text: str) -> Sequence[int]:
        """Encode text into token IDs."""
        result = tokenize(
            content=text,
            add_special=self.add_special_tokens,
            model=self.model,
            use_server=False,
            auto_fallback=True,
        )
        return result["tokens"]

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode token IDs back to text."""
        result = detokenize(
            tokens=list(tokens),
            model=self.model,
            use_server=False,
            skip_special_tokens=self.skip_special_tokens,
        )
        return result["content"]

    def tokenize(self, text: str) -> Sequence[str]:
        """Tokenize text into string tokens (pieces)."""
        result = tokenize(
            content=text,
            add_special=self.add_special_tokens,
            with_pieces=True,
            model=self.model,
            use_server=False,
            auto_fallback=True,
        )
        pieces = []
        for tok in result["tokens"]:
            if isinstance(tok, dict):
                piece = tok.get("piece", "")
                pieces.append(piece if isinstance(piece, str) else str(piece))
            else:
                decoded = self.decode([tok])
                pieces.append(decoded)
        return pieces

    def count_tokens(self, text: str) -> int:
        """Count tokens — uses optimized path from token_utils."""
        return count_tokens(
            content=text,
            add_special=self.add_special_tokens,
            model=self.model,
            use_server=False,
            auto_fallback=True,
        )

    def count_tokens_batch(self, texts: Sequence[str]) -> Sequence[int]:
        """Batch count tokens efficiently."""
        return [self.count_tokens(text) for text in texts]

    def encode_batch(self, texts: Sequence[str]) -> Sequence[Sequence[int]]:
        """Batch encode texts."""
        return [self.encode(text) for text in texts]

    def decode_batch(self, token_sequences: Sequence[Sequence[int]]) -> Sequence[str]:
        """Batch decode token sequences."""
        return [self.decode(tokens) for tokens in token_sequences]
