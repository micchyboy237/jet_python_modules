from typing import List

from jet.logger import logger
from jet.wordnet.words import get_words
from tokenizers import Encoding, Tokenizer


# Mock tokenizer for consistent token counting
class MockTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        words = get_words(text)

        class MockEncoding:
            def __init__(self, tokens: List[int], text: str):
                self.ids = tokens
                self.offsets = [(i, i + len(word))
                                for i, word in enumerate(words)]
        # Simple tokenization: split by whitespace and count words
        tokens = list(range(len(words)))
        logger.debug(
            f"MockTokenizer: Encoding text='{text}', tokens={len(tokens)}")
        return MockEncoding(tokens, text)
