from typing import List, Union, Callable, Optional
from jet.llm.mlx.mlx_types import ModelType
from jet.llm.mlx.models import resolve_model_value
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizer, BatchEncoding
from jet.logger import logger

# Standalone reusable functions


def load_model(model_name: ModelType):
    model_id = resolve_model_value(model_name)
    model = SentenceTransformer(model_id)
    return model


def generate_embeddings(
    model_name: ModelType,
    text: Union[str, List[str]]
) -> Union[List[float], List[List[float]]]:
    """Get embeddings for a single string or list of strings using the provided model."""
    _model = load_model(model_name)

    encoded: np.ndarray = _model.encode(
        [text] if isinstance(text, str) else text, batch_size=8
    )
    return encoded[0].tolist() if isinstance(text, str) else [vec.tolist() for vec in encoded]


def get_embedding_function(
    model_name: ModelType
) -> Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]]:
    """Returns an embedding function for the specified model name."""
    _model = load_model(model_name)

    def embed(text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        encoded: np.ndarray = _model.encode(
            [text] if isinstance(text, str) else text, batch_size=8
        )
        return encoded[0].tolist() if isinstance(text, str) else [vec.tolist() for vec in encoded]

    return embed


def get_token_counts(
    model_name: ModelType,
    texts: Union[str, List[str]]
) -> Union[int, List[int]]:
    """Returns token counts using batch_encode_plus."""
    _model = load_model(model_name)
    tokenizer = _model.tokenizer
    single = isinstance(texts, str)
    texts = [texts] if single else texts

    encoded: BatchEncoding = tokenizer.batch_encode_plus(
        texts, add_special_tokens=True
    )
    counts = [len(input_ids) for input_ids in encoded['input_ids']]
    return counts[0] if single else counts


def get_token_counts_alt(
    model_name: ModelType,
    texts: Union[str, List[str]]
) -> Union[int, List[int]]:
    """Returns token counts using tokenizer directly."""
    _model = load_model(model_name)
    tokenizer = _model.tokenizer
    single = isinstance(texts, str)
    texts = [texts] if single else texts

    encoded_ids: List[List[int]] = tokenizer(
        texts, add_special_tokens=True
    )['input_ids']
    counts = [len(input_ids) for input_ids in encoded_ids]
    return counts[0] if single else counts


def tokenize(
    model_name: ModelType,
    text: Union[str, List[str]]
) -> Union[List[int], List[List[int]]]:
    """Tokenizes a single string or list of strings, returns token IDs."""
    _model = load_model(model_name)
    tokenizer = _model.tokenizer
    encoded = tokenizer(text, add_special_tokens=True)
    return encoded['input_ids']


def tokenize_strings(
    model_name: ModelType,
    text: Union[str, List[str]]
) -> Union[List[str], List[List[str]]]:
    """Returns tokens as readable strings using convert_ids_to_tokens."""
    _model = load_model(model_name)
    tokenizer = _model.tokenizer
    ids = tokenize(model_name, text)
    if isinstance(text, str):
        return tokenizer.convert_ids_to_tokens(ids)
    return [tokenizer.convert_ids_to_tokens(seq) for seq in ids]


def get_tokenizer_fn(
    model_name: ModelType
) -> Callable[[Union[str, List[str]]], Union[List[int], List[List[int]]]]:
    """Returns a pre-configured tokenizer function with special tokens added."""
    _model = load_model(model_name)
    tokenizer = _model.tokenizer

    def token_fn(text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        return tokenizer(text, add_special_tokens=True)['input_ids']
    return token_fn


class SentenceEmbedding:
    def __init__(self, model_name: ModelType):
        self.model_name = model_name
        self.model: SentenceTransformer = load_model(model_name)
        self.tokenizer: PreTrainedTokenizer = self.model.tokenizer
        logger.info(f"Model '{model_name}' and tokenizer initialized.")

    def generate_embeddings(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Get embeddings using the instance model."""
        return generate_embeddings(self.model_name, text)

    def get_embedding_function(self, model_name: Optional[str] = None) -> Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]]:
        """Returns an embedding function for the specified or instance model name."""
        return get_embedding_function(model_name or self.model_name)

    def get_token_counts(self, texts: Union[str, List[str]]) -> Union[int, List[int]]:
        """Returns token counts using batch_encode_plus."""
        return get_token_counts(self.model_name, texts)

    def get_token_counts_alt(self, texts: Union[str, List[str]]) -> Union[int, List[int]]:
        """Returns token counts using tokenizer directly."""
        return get_token_counts_alt(self.model_name, texts)

    def tokenize(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """Tokenizes a single string or list of strings, returns token IDs."""
        return tokenize(self.model_name, text)

    def tokenize_strings(self, text: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        """Returns tokens as readable strings."""
        return tokenize_strings(self.model_name, text)

    def get_tokenizer_fn(self, model_name: Optional[str] = None) -> Callable[[Union[str, List[str]]], Union[List[int], List[List[int]]]]:
        """Returns a pre-configured tokenizer function."""
        return get_tokenizer_fn(model_name or self.model_name)


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

    fn = util.get_tokenizer_fn()
    print(f"get_tokenizer_fn single: {fn(text)}")
    print(f"get_tokenizer_fn list: {fn(texts)}")

    logger.info("Using get_embedding_function with 'all-MiniLM-L6-v2'")
    embed_fn = util.get_embedding_function('all-MiniLM-L6-v2')
    emb_fn_single = embed_fn(text)
    emb_fn_list = embed_fn(texts)
    print(f"[get_embedding_function] Single length: {len(emb_fn_single)}")
    print(
        f"[get_embedding_function] List shape: ({len(emb_fn_list)}, {len(emb_fn_list[0])})")
