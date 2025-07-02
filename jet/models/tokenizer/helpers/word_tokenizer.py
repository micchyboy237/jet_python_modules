from typing import Dict, List, Optional, Union
from jet.models.tokenizer.base import TokenizerWrapper
import re
import logging

from jet.wordnet.words import get_words

logger = logging.getLogger(__name__)


class WordTokenizer(TokenizerWrapper):
    def __init__(self, **kwargs):
        # Initialize basic vocabulary with special tokens
        self._vocab: Dict[str, int] = {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
        }
        self._reverse_vocab: Dict[int, str] = {
            v: k for k, v in self._vocab.items()}
        self._added_tokens: List[str] = []
        self.model_max_length: int = kwargs.get("max_length", 512)
        self._pad_token: str = "<pad>"
        self._unk_token: str = "<unk>"
        self._bos_token: str = "<s>"
        self._eos_token: str = "</s>"
        pad_token_id: int = self._vocab.get(self._pad_token, 0)

        # Initialize TokenizerWrapper with self as the tokenizer
        super().__init__(
            tokenizer=self,
            remove_pad_tokens=kwargs.get("remove_pad_tokens", False),
            add_special_tokens=kwargs.get("add_special_tokens", True),
            pad_token_id=pad_token_id,
        )

    def tokenize(self, text: str) -> List[str]:
        """Tokenize input text into words and handle special tokens."""
        logger.debug(f"Tokenizing text: {text}")
        words = get_words(text)
        return words

    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode a single text string into token IDs."""
        logger.debug(f"Encoding text: {text}, kwargs: {kwargs}")
        tokens = self.tokenize(text)
        token_ids = [self._vocab.get(
            token, self._vocab[self._unk_token]) for token in tokens]
        if kwargs.get("add_special_tokens", self.add_special_tokens):
            token_ids = [self._vocab[self._bos_token]] + \
                token_ids + [self._vocab[self._eos_token]]
        if self.remove_pad_tokens:
            token_ids = [tid for tid in token_ids if tid !=
                         self._vocab[self._pad_token]]
        return token_ids

    def encode_batch(self, texts: List[str], **kwargs) -> List[List[int]]:
        """Encode a batch of texts into token IDs."""
        logger.debug(f"Encoding batch texts: {texts}, kwargs: {kwargs}")
        return [self.encode(text, **kwargs) for text in texts]

    def decode(self, token_ids: Union[List[int], List[List[int]]], **kwargs) -> Union[str, List[str]]:
        """Decode token IDs back to text."""
        logger.debug(f"Decoding token IDs: {token_ids}, kwargs: {kwargs}")
        skip_special = kwargs.get(
            "skip_special_tokens", self.add_special_tokens)
        if isinstance(token_ids[0], int):
            tokens = [self._reverse_vocab.get(tid, self._unk_token) for tid in token_ids if not (
                skip_special and tid in {self._vocab[self._bos_token], self._vocab[self._eos_token]})]
            return " ".join(tokens)
        return [self.decode(ids, **kwargs) for ids in token_ids]

    def convert_ids_to_tokens(self, token_ids: Union[List[int], List[List[int]]], **kwargs) -> Union[List[str], List[List[str]]]:
        """Convert token IDs to their string representations."""
        logger.debug(
            f"Converting token IDs to tokens: {token_ids}, kwargs: {kwargs}")
        skip_special = kwargs.get(
            "skip_special_tokens", self.add_special_tokens)
        if isinstance(token_ids[0], int):
            return [self._reverse_vocab.get(tid, self._unk_token) for tid in token_ids if not (skip_special and tid in {self._vocab[self._bos_token], self._vocab[self._eos_token]})]
        return [self.convert_ids_to_tokens(ids, **kwargs) for ids in token_ids]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert list of tokens back to a string."""
        logger.debug(f"Converting tokens to string: {tokens}")
        return " ".join(tokens)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert list of tokens to their corresponding IDs."""
        logger.debug(f"Converting tokens to IDs: {tokens}")
        return [self._vocab.get(token, self._vocab[self._unk_token]) for token in tokens]

    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary dictionary."""
        return self._vocab

    def _add_tokens(self, new_tokens: List[str], special_tokens: bool = False) -> int:
        """Add new tokens to the vocabulary."""
        logger.debug(
            f"Adding tokens: {new_tokens}, special_tokens: {special_tokens}")
        added_count = 0
        for token in new_tokens:
            if token not in self._vocab:
                token_id = len(self._vocab)
                self._vocab[token] = token_id
                self._reverse_vocab[token_id] = token
                self._added_tokens.append(token)
                added_count += 1
        return added_count

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self._vocab)

    @property
    def total_vocab_size(self) -> int:
        """Return the total size of the vocabulary."""
        return len(self._vocab)

    @property
    def pad_token(self) -> str:
        """Return the padding token."""
        return self._pad_token

    @property
    def unk_token(self) -> str:
        """Return the unknown token."""
        return self._unk_token

    @property
    def unk_token_id(self) -> int:
        """Return the unknown token ID."""
        return self._vocab[self._unk_token]

    @property
    def bos_token(self) -> str:
        """Return the beginning-of-sequence token."""
        return self._bos_token

    @property
    def bos_token_id(self) -> int:
        """Return the beginning-of-sequence token ID."""
        return self._vocab[self._bos_token]

    @property
    def eos_token(self) -> str:
        """Return the end-of-sequence token."""
        return self._eos_token

    @property
    def eos_token_id(self) -> int:
        """Return the end-of-sequence token ID."""
        return self._vocab[self._eos_token]
