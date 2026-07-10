import os
from typing import Any, Dict, List, Optional, Union, TypedDict
import requests
from openai import OpenAI
from jet.adapters.llama_cpp.config import LLM_BASE_URL
from jet.adapters.llama_cpp.factory import get_llm_client

# Environment-aware base URLs (reuses config + supports new embed host export)
LLM_BASE_URL_ENV = (
    os.getenv("LLAMA_CPP_LLM_HOST") or LLM_BASE_URL
)  # No /v1 - for native endpoints
LLM_V1_URL_ENV = os.getenv(
    "LLAMA_CPP_LLM_URL"
)  # With /v1 - for OpenAI-compatible endpoints
EMBED_BASE_URL_ENV = os.getenv("LLAMA_CPP_EMBED_HOST") or os.getenv(
    "LLAMA_CPP_EMBED_URL"
)


class Token(TypedDict):
    id: int
    piece: Union[str, List[int]]


class TokenizeResponse(TypedDict):
    tokens: List[Union[int, Token]]


class DetokenizeResponse(TypedDict):
    content: str


class ModelInfo(TypedDict):
    id: str
    object: str
    created: int
    owned_by: str
    meta: Optional[Dict[str, Any]]


class ModelsResponse(TypedDict):
    object: str
    data: List[ModelInfo]


def _get_llm_base_url(override: Optional[str] = None) -> str:
    """Return LLM server base URL (NO /v1) for native llama.cpp endpoints.

    Used by endpoints like /tokenize, /detokenize, /completion, /embedding, /health
    """
    base = override or LLM_BASE_URL_ENV
    return base.rstrip("/") if base else "http://localhost:8080"


def _get_embed_base_url(override: Optional[str] = None) -> str:
    """Return Embed server base URL (supports LLAMA_CPP_EMBED_HOST export)."""
    base = override or EMBED_BASE_URL_ENV
    return base.rstrip("/") if base else "http://localhost:8081"


def tokenize(
    content: str,
    add_special: bool = False,
    parse_special: bool = True,
    with_pieces: bool = False,
    base_url: Optional[str] = None,
) -> TokenizeResponse:
    """
    Tokenize text via llama.cpp /tokenize endpoint.

    Note: This is a NATIVE llama.cpp endpoint (no /v1 prefix).

    Args:
        content: Text to tokenize.
        add_special: Insert special tokens (BOS etc.).
        parse_special: Parse special tokens vs treat as text.
        with_pieces: Return token pieces (for debugging).
        base_url: Override server URL (should NOT include /v1).

    Returns:
        Tokenization result.
    """
    url = f"{_get_llm_base_url(base_url)}/tokenize"
    payload: Dict[str, Any] = {
        "content": content,
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
) -> DetokenizeResponse:
    """
    Convert token IDs back to text via /detokenize.

    Note: This is a NATIVE llama.cpp endpoint (no /v1 prefix).

    Args:
        tokens: Token ID list.
        base_url: Override server URL (should NOT include /v1).

    Returns:
        Detokenized text.
    """
    url = f"{_get_llm_base_url(base_url)}/detokenize"
    payload: Dict[str, Any] = {"tokens": tokens}
    response = requests.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()  # type: ignore


def count_tokens(
    content: Union[str, List[Union[int, str, List[int]]]],
    add_special: bool = False,
    base_url: Optional[str] = None,
) -> int:
    """
    Count tokens in content (string or token list).

    Uses /tokenize for strings, or simply counts list length.

    Args:
        content: Text or token sequence.
        add_special: Add special tokens when tokenizing strings.
        base_url: Override server URL (should NOT include /v1).

    Returns:
        Token count.
    """
    if isinstance(content, str):
        result = tokenize(
            content,
            add_special=add_special,
            base_url=base_url,
        )
        return len(result["tokens"])
    if isinstance(content, list):
        return len(content)
    return 0


def get_models(base_url: Optional[str] = None) -> ModelsResponse:
    """
    Get loaded model(s) via OpenAI-compatible /v1/models.

    Note: This uses the OpenAI client which needs the /v1 prefix.

    Args:
        base_url: Override server URL (should include /v1).

    Returns:
        Models list response.
    """
    client: OpenAI = get_llm_client()
    if base_url:
        # Temporary override for this call
        original_base = client.base_url
        client.base_url = base_url  # type: ignore[attr-defined]
        try:
            models = client.models.list()
            return models.model_dump()  # type: ignore[attr-defined]
        finally:
            client.base_url = original_base  # type: ignore[attr-defined]
    models = client.models.list()
    return models.model_dump()  # type: ignore[attr-defined]


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

    # Get models (v1 endpoint - uses /v1/models)
    models = get_models()
    for model in models["data"]:
        print(f"\nModel: {model['id']}")
        for key, value in model.items():
            if key != "id":  # Already printed above
                print(f"  {key}: {value}")
