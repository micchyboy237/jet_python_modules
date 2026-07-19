# jet.adapters.llama_cpp.token_utils
from typing import Any, Dict, List, Optional, TypedDict, Union

import requests
from jet.adapters.llama_cpp.config import EMBED_MODEL, LLM_MODEL
from jet.adapters.llama_cpp.model_utils import ServerType, get_llama_cpp_base_url
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS, LLAMACPP_LLM_KEYS


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
    base_url: Optional[str] = None,
    model: Optional[LLAMACPP_LLM_KEYS | LLAMACPP_EMBED_KEYS] = None,
    server: ServerType = "llm",
) -> TokenizeResponse:
    """
    Tokenize text via llama.cpp /tokenize endpoint.

    Note: This is a NATIVE llama.cpp endpoint (no /v1 prefix).
    """
    if model is None:
        if server == "llm":
            model: LLAMACPP_LLM_KEYS = LLM_MODEL
        else:
            model: LLAMACPP_EMBED_KEYS = EMBED_MODEL

    url = f"{get_llama_cpp_base_url(server=server, override=base_url)}/tokenize"
    payload: Dict[str, Any] = {
        "content": content,
        "model": model,
        "add_special": add_special,
        "parse_special": parse_special,
        "with_pieces": with_pieces,
    }
    response = requests.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()  # type: ignore


def detokenize(
    tokens: List[int],
    base_url: Optional[str] = None,
    model: Optional[LLAMACPP_LLM_KEYS | LLAMACPP_EMBED_KEYS] = None,
    server: ServerType = "llm",
) -> DetokenizeResponse:
    """
    Convert token IDs back to text via /detokenize.
    """
    if model is None:
        if server == "llm":
            model: LLAMACPP_LLM_KEYS = LLM_MODEL
        else:
            model: LLAMACPP_EMBED_KEYS = EMBED_MODEL

    url = f"{get_llama_cpp_base_url(server=server, override=base_url)}/detokenize"
    payload: Dict[str, Any] = {
        "model": model,
        "tokens": tokens,
    }
    response = requests.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()  # type: ignore


def count_tokens(
    content: Union[str, List[Union[int, str, List[int]]]],
    add_special: bool = False,
    base_url: Optional[str] = None,
    model: Optional[LLAMACPP_LLM_KEYS | LLAMACPP_EMBED_KEYS] = None,
    server: ServerType = "llm",
) -> int:
    """
    Count tokens in content (string or token list).
    """
    if isinstance(content, str):
        result = tokenize(
            content,
            add_special=add_special,
            base_url=base_url,
            model=model,
            server=server,
        )
        return len(result["tokens"])
    if isinstance(content, list):
        return len(content)
    return 0


if __name__ == "__main__":
    # Tokenize (native endpoint - no /v1)
    tokens_resp = tokenize("Hello world!", add_special=True, with_pieces=True)
    print("Tokens:", tokens_resp["tokens"])

    # Detokenize (native endpoint - no /v1)
    text_resp = detokenize([123, 456, 789])
    print("Detokenized:", text_resp["content"])

    # Count tokens (native endpoint - no /v1)
    num_tokens = count_tokens("This is a test prompt.")
    print("Token count:", num_tokens)
