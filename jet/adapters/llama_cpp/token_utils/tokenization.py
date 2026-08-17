from typing import List, TypedDict, Union

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.logger import logger

from .tokenizer_management import get_tokenizer


class Token(TypedDict):
    id: int
    piece: Union[str, List[int]]


class TokenizeResponse(TypedDict):
    tokens: List[Union[int, Token]]


class DetokenizeResponse(TypedDict):
    content: str


def tokenize(
    content: str,
    add_special: bool = False,
    parse_special: bool = True,
    with_pieces: bool = False,
    base_url: str | None = None,
    model: str | None = None,
    use_server: bool = False,
    auto_fallback: bool = True,  # NEW
) -> TokenizeResponse:
    """
    Tokenize using server if available, otherwise fallback to local.
    """
    if model is None:
        model = LLM_MODEL

    if use_server:
        from .server_health import is_server_available

        if is_server_available(base_url):
            from .server_interaction import tokenize as server_tokenize

            return server_tokenize(
                content, add_special, parse_special, with_pieces, base_url, model
            )
        else:
            logger.warning(f"Server not available, using local tokenizer")
            if not auto_fallback:
                raise ConnectionError(
                    f"Server not available and auto_fallback disabled"
                )

    # Local processing
    tokenizer = get_tokenizer(model)
    token_ids = tokenizer.encode(content, add_special_tokens=add_special)
    if with_pieces:
        tokens: List[Union[int, Token]] = []
        for tid in token_ids:
            piece = tokenizer.decode([tid])
            tokens.append(
                {"id": tid, "piece": piece}
            )  # Use dict for TypedDict compatibility
        return TokenizeResponse(tokens=tokens)
    else:
        return TokenizeResponse(tokens=token_ids)


def detokenize(
    tokens: List[int],
    base_url: str | None = None,
    model: str | None = None,
    use_server: bool = False,
    skip_special_tokens: bool = True,
) -> DetokenizeResponse:
    """
    Convert token IDs back to text using local tokenizer by default, or /detokenize endpoint.
    """
    if model is None:
        model = LLM_MODEL
    if use_server:
        from .server_interaction import detokenize as server_detokenize

        return server_detokenize(tokens, base_url, model, skip_special_tokens)
    logger.debug(f"Using local detokenizer for model: {model}")
    tokenizer = get_tokenizer(model)
    content = tokenizer.decode(
        tokens,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=True,
    )
    return DetokenizeResponse(content=content)
