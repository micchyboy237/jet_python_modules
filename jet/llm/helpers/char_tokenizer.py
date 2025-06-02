from transformers import PreTrainedTokenizer
from typing import Dict, List


# Custom character-based tokenizer for concatenated characters
class CharTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._vocab = {chr(i): i for i in range(128)}  # Basic ASCII vocab
        self._added_tokens = []
        self.model_max_length = kwargs.get(
            'max_length', 512)  # Default max length
        self._pad_token = "<pad>"
        self._pad_token_id = self._vocab.get(self._pad_token, 0)

    def tokenize(self, text: str) -> List[str]:
        return list(text)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab

    def _add_tokens(self, new_tokens, special_tokens=False):
        for token in new_tokens:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)
                self._added_tokens.append(token)
        return len(new_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def total_vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def pad_token(self) -> str:
        return self._pad_token

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id
