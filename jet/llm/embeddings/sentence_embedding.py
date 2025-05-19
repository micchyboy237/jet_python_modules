from typing import List, Union, Callable, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizer, BatchEncoding
from jet.logger import logger


class SentenceEmbedding:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.tokenizer: PreTrainedTokenizer = self.model.tokenizer
        logger.info(f"Model '{model_name}' and tokenizer initialized.")

    def generate_embeddings(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Get embeddings for a single string or list of strings using the initialized model."""
        encoded: np.ndarray = self.model.encode(
            [text] if isinstance(text, str) else text, batch_size=8
        )
        return encoded[0].tolist() if isinstance(text, str) else [vec.tolist() for vec in encoded]

    def get_embedding_function(self, model_name: Optional[str] = None) -> Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]]:
        """Returns an embedding function using the specified or the instance model name."""
        _model = SentenceTransformer(model_name or self.model_name)

        def embed(text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
            encoded: np.ndarray = _model.encode(
                [text] if isinstance(text, str) else text, batch_size=8
            )
            return encoded[0].tolist() if isinstance(text, str) else [vec.tolist() for vec in encoded]

        return embed

    def get_token_counts(self, texts: Union[str, List[str]]) -> Union[int, List[int]]:
        """Returns token counts using batch_encode_plus."""
        single = isinstance(texts, str)
        texts = [texts] if single else texts

        encoded: BatchEncoding = self.tokenizer.batch_encode_plus(
            texts, add_special_tokens=True
        )
        counts = [len(input_ids) for input_ids in encoded['input_ids']]
        return counts[0] if single else counts

    def get_token_counts_alt(self, texts: Union[str, List[str]]) -> Union[int, List[int]]:
        """Returns token counts using tokenizer directly."""
        single = isinstance(texts, str)
        texts = [texts] if single else texts

        encoded_ids: List[List[int]] = self.tokenizer(
            texts, add_special_tokens=True
        )['input_ids']
        counts = [len(input_ids) for input_ids in encoded_ids]
        return counts[0] if single else counts

    def tokenize(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """Tokenizes a single string or list of strings, returns token IDs."""
        encoded = self.tokenizer(text, add_special_tokens=True)
        return encoded['input_ids']

    def tokenize_strings(self, text: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        """Returns tokens as readable strings using convert_ids_to_tokens."""
        ids = self.tokenize(text)
        if isinstance(text, str):
            return self.tokenizer.convert_ids_to_tokens(ids)
        return [self.tokenizer.convert_ids_to_tokens(seq) for seq in ids]

    def get_tokenize_fn(self) -> Callable[[Union[str, List[str]]], Union[List[int], List[List[int]]]]:
        """Returns a pre-configured tokenizer function with special tokens added."""

        def token_fn(text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
            return self.tokenizer(text, add_special_tokens=True)['input_ids']

        return token_fn


if __name__ == '__main__':
    # Initialize instance
    util = SentenceEmbedding('all-MiniLM-L6-v2')

    text = "This is the first sample sentence."
    texts = [
        "This is the first sample sentence.",
        "Another sentence to encode and count tokens.",
        "Short text."
    ]

    emb_single = util.generate_embeddings(text)
    emb_list = util.generate_embeddings(texts)
    print(f"Single embedding length: {len(emb_single)}")
    print(f"Batch embedding shape: ({len(emb_list)}, {len(emb_list[0])})")

    count1 = util.get_token_counts(text)
    count2 = util.get_token_counts(texts)
    alt_count1 = util.get_token_counts_alt(text)
    alt_count2 = util.get_token_counts_alt(texts)

    logger.info("Token counts with batch_encode_plus:")
    logger.success(f"Single: {count1}")
    for t, c in zip(texts, count2):
        logger.success(f"Text: {t}\nToken count: {c}\n")

    logger.info("Token counts with direct tokenizer:")
    logger.success(f"Single: {alt_count1}")
    for t, c in zip(texts, alt_count2):
        logger.success(f"Text: {t}\nToken count: {c}\n")

    token_ids_1 = util.tokenize(text)
    token_ids_2 = util.tokenize(texts)
    print(f"Token IDs (single): {token_ids_1}")
    print(f"Token IDs (list): {token_ids_2}")

    readable_1 = util.tokenize_strings(text)
    readable_2 = util.tokenize_strings(texts)
    print(f"Tokens (single): {readable_1}")
    print(f"Tokens (list): {readable_2}")

    fn = util.get_tokenize_fn()
    print(f"get_tokenize_fn single: {fn(text)}")
    print(f"get_tokenize_fn list: {fn(texts)}")

    logger.info("Using get_embedding_function with 'all-MiniLM-L6-v2'")
    embed_fn = util.get_embedding_function('all-MiniLM-L6-v2')
    emb_fn_single = embed_fn(text)
    emb_fn_list = embed_fn(texts)
    print(f"[get_embedding_function] Single length: {len(emb_fn_single)}")
    print(
        f"[get_embedding_function] List shape: ({len(emb_fn_list)}, {len(emb_fn_list[0])})")
