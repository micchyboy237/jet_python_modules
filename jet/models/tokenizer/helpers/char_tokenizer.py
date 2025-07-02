from typing import Dict, List, Optional, Union
from jet.models.tokenizer.base import TokenizerWrapper
import logging

logger = logging.getLogger(__name__)


class CharTokenizer(TokenizerWrapper):
    def __init__(self, **kwargs):
        # Initialize basic ASCII vocabulary
        self._vocab: Dict[str, int] = {chr(i): i for i in range(128)}
        self._reverse_vocab: Dict[int, str] = {
            v: k for k, v in self._vocab.items()}
        self._added_tokens: List[str] = []
        self._pad_token: str = "<pad>"
        self.model_max_length: int = kwargs.get("max_length", 512)
        pad_token_id: int = self._vocab.get(self._pad_token, 0)

        # Initialize TokenizerWrapper with self as the tokenizer
        super().__init__(
            tokenizer=self,
            remove_pad_tokens=kwargs.get("remove_pad_tokens", False),
            add_special_tokens=kwargs.get("add_special_tokens", True),
            pad_token_id=pad_token_id,
        )

    def tokenize(self, text: str) -> List[str]:
        """Tokenize input text into individual characters."""
        logger.debug(f"Tokenizing text: {text}")
        return list(text)

    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode a single text string into token IDs."""
        logger.debug(f"Encoding text: {text}, kwargs: {kwargs}")
        tokens = self.tokenize(text)
        token_ids = [self._vocab.get(token, self.pad_token_id)
                     for token in tokens]
        if kwargs.get("add_special_tokens", self.add_special_tokens):
            token_ids = [self._vocab.get("<s>", 2)] + \
                token_ids + [self._vocab.get("</s>", 3)]
        if self.remove_pad_tokens:
            token_ids = [tid for tid in token_ids if tid != self.pad_token_id]
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
            tokens = [self._reverse_vocab.get(tid, "<unk>") for tid in token_ids if not (
                skip_special and tid in {2, 3})]
            return "".join(tokens)
        return [self.decode(ids, **kwargs) for ids in token_ids]

    def convert_ids_to_tokens(self, token_ids: Union[List[int], List[List[int]]], **kwargs) -> Union[List[str], List[List[str]]]:
        """Convert token IDs to their string representations."""
        logger.debug(
            f"Converting token IDs to tokens: {token_ids}, kwargs: {kwargs}")
        skip_special = kwargs.get(
            "skip_special_tokens", self.add_special_tokens)
        if isinstance(token_ids[0], int):
            return [self._reverse_vocab.get(tid, "<unk>") for tid in token_ids if not (skip_special and tid in {2, 3})]
        return [self.convert_ids_to_tokens(ids, **kwargs) for ids in token_ids]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert list of tokens back to a string."""
        logger.debug(f"Converting tokens to string: {tokens}")
        return "".join(tokens)

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
