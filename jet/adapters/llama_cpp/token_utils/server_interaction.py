from typing import Any, Dict, List, Union

import requests
from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.model_utils import get_llama_cpp_base_url
from jet.logger import logger


def tokenize(
    content: str,
    add_special: bool = False,
    parse_special: bool = True,
    with_pieces: bool = False,
    base_url: str | None = None,
    model: str | None = None,
) -> Dict[str, Any]:
    """
    Tokenize text using llama.cpp /tokenize endpoint.
    """
    if model is None:
        model = LLM_MODEL
    url = f"{get_llama_cpp_base_url(override=base_url)}/tokenize"
    payload: Dict[str, Any] = {
        "content": content,
        "model": model,
        "add_special": add_special,
        "parse_special": parse_special,
        "with_pieces": with_pieces,
    }
    logger.debug(f"Tokenizing via server: {url} (model={model})")
    response = requests.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()


def detokenize(
    tokens: List[int],
    base_url: str | None = None,
    model: str | None = None,
    skip_special_tokens: bool = True,
) -> Dict[str, Any]:
    """
    Convert token IDs back to text using /detokenize endpoint.
    """
    if model is None:
        model = LLM_MODEL
    url = f"{get_llama_cpp_base_url(override=base_url)}/detokenize"
    payload: Dict[str, Any] = {
        "model": model,
        "tokens": tokens,
    }
    logger.debug(f"Detokenizing via server: {url} (model={model})")
    response = requests.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()


def count_tokens_raw(
    input_text: str,
    model: str | None = None,
    base_url: str | None = None,
    **kwargs,
) -> int:
    """
    Count tokens using /v1/responses/input_tokens endpoint.
    """
    if model is None:
        model = LLM_MODEL
    base = get_llama_cpp_base_url(override=base_url)
    url = f"{base}/v1/responses/input_tokens"
    payload: Dict[str, Any] = {
        "model": model,
        "input": input_text,
        **kwargs,
    }
    logger.debug(f"Counting raw tokens via server: {url}")
    response = requests.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    result = response.json()
    return result["input_tokens"]


def count_tokens(
    content: Any,
    add_special: bool = False,
    base_url: str | None = None,
    model: str | None = None,
    tools: List[Dict[str, Any]] | None = None,
    prevent_total: bool = False,
    **kwargs,
) -> Union[int, List[int]]:
    """
    Count tokens using llama.cpp server endpoints.
    """
    if model is None:
        model = LLM_MODEL
    if isinstance(content, str):
        try:
            result = count_tokens_raw(
                content,
                model=model,
                base_url=base_url,
                **kwargs,
            )
            return result
        except Exception as e:
            logger.warning(f"Raw token count failed, falling back to tokenize: {e}")
            result = tokenize(
                content,
                add_special=add_special,
                base_url=base_url,
                model=model,
            )
            return len(result["tokens"])
    if isinstance(content, list):
        if not content:
            return [] if prevent_total else 0
        if content and all(
            isinstance(item, dict) and "role" in item and "content" in item
            for item in content
        ):
            try:
                result = count_chat_tokens(
                    content,
                    model=model,
                    base_url=base_url,
                    tools=tools,
                    **kwargs,
                )
                return result["input_tokens"]
            except Exception as e:
                logger.warning(
                    f"Chat token count failed, falling back to tokenize: {e}"
                )
                combined = " ".join(msg.get("content", "") for msg in content)
                result = tokenize(
                    combined,
                    add_special=add_special,
                    base_url=base_url,
                    model=model,
                )
                return len(result["tokens"])
        elif content and all(isinstance(item, str) for item in content):
            token_counts = []
            for text in content:
                try:
                    result = count_tokens_raw(
                        text,
                        model=model,
                        base_url=base_url,
                        **kwargs,
                    )
                    token_counts.append(result)
                except Exception as e:
                    logger.warning(
                        f"Raw token count failed for string, falling back: {e}"
                    )
                    result = tokenize(
                        text,
                        add_special=add_special,
                        base_url=base_url,
                        model=model,
                    )
                    token_counts.append(len(result["tokens"]))
            return token_counts if prevent_total else sum(token_counts)
        else:
            return len(content)
    logger.warning(f"Unsupported content type: {type(content)}")
    return 0


def count_chat_tokens(
    messages: List[Dict[str, str]],
    model: str | None = None,
    base_url: str | None = None,
    tools: List[Dict[str, Any]] | None = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Count tokens using /v1/chat/completions/input_tokens endpoint.
    """
    if model is None:
        model = LLM_MODEL
    base = get_llama_cpp_base_url(override=base_url)
    url = f"{base}/v1/chat/completions/input_tokens"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        **kwargs,
    }
    if tools:
        payload["tools"] = tools
    logger.debug(f"Counting chat tokens via server: {url}")
    response = requests.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    result = response.json()
    return result
