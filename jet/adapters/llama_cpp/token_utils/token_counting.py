from typing import Any, Dict, List, Literal, TypedDict, Union, overload

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.logger import logger

from .tokenizer_management import get_tokenizer


class InputTokensResponse(TypedDict):
    input_tokens: int
    object: str


@overload
def count_tokens(
    content: Union[str, List[Union[int, str, List[int], Dict[str, str]]]],
    add_special: bool = False,
    base_url: str | None = None,
    model: str | None = None,
    tools: List[Dict[str, Any]] | None = None,
    use_server: bool = False,
    prevent_total: Literal[False] = False,
    auto_fallback: bool = True,
    **kwargs,
) -> int: ...


@overload
def count_tokens(
    content: Union[str, List[Union[int, str, List[int], Dict[str, str]]]],
    add_special: bool = False,
    base_url: str | None = None,
    model: str | None = None,
    tools: List[Dict[str, Any]] | None = None,
    use_server: bool = False,
    prevent_total: Literal[True] = ...,
    auto_fallback: bool = True,
    **kwargs,
) -> List[int]: ...


def count_tokens(
    content: Union[str, List[Union[int, str, List[int], Dict[str, str]]]],
    add_special: bool = False,
    base_url: str | None = None,
    model: str | None = None,
    tools: List[Dict[str, Any]] | None = None,
    use_server: bool = False,
    prevent_total: Literal[False] = False,
    auto_fallback: bool = True,  # NEW: auto-fallback to local
    **kwargs,
) -> Union[int, List[int]]:
    """
    Count tokens using local tokenizer by default, with intelligent detection.
    If use_server=True and server is unavailable, falls back to local if auto_fallback=True.
    """
    if model is None:
        model = LLM_MODEL

    if use_server:
        # Check if server is available
        from .server_health import is_server_available

        if is_server_available(base_url):
            from .server_interaction import count_tokens as server_count_tokens

            return server_count_tokens(
                content, add_special, base_url, model, tools, prevent_total, **kwargs
            )
        else:
            logger.warning(f"Server not available at {base_url}, falling back to local")
            if not auto_fallback:
                raise ConnectionError(
                    f"Server not available and auto_fallback disabled"
                )
            # Fall through to local processing

    tokenizer = get_tokenizer(model)
    if isinstance(content, str):
        return len(tokenizer.encode(content, add_special_tokens=add_special))
    if isinstance(content, list):
        if not content:
            return [] if prevent_total else 0
        if content and all(
            isinstance(item, dict) and "role" in item and "content" in item
            for item in content
        ):
            try:
                token_res = tokenizer.apply_chat_template(
                    content,
                    tools=tools,
                    tokenize=True,
                    add_generation_prompt=False,
                )
                return len(token_res["input_ids"])
            except Exception as e:
                logger.warning(
                    f"Chat template failed, falling back to concatenation: {e}"
                )
                combined = " ".join(msg.get("content", "") for msg in content)
                return len(tokenizer.encode(combined, add_special_tokens=add_special))
        elif content and all(isinstance(item, str) for item in content):
            token_counts = [
                len(tokenizer.encode(text, add_special_tokens=add_special))
                for text in content
            ]
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
    use_server: bool = False,
    **kwargs,
) -> InputTokensResponse:
    """
    Count tokens using local tokenizer by default, or /v1/chat/completions/input_tokens endpoint.
    """
    if model is None:
        model = LLM_MODEL
    if use_server:
        from .server_interaction import count_chat_tokens as server_count_chat_tokens

        return server_count_chat_tokens(messages, model, base_url, tools, **kwargs)
    tokenizer = get_tokenizer(model)
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
        )
        return InputTokensResponse(
            input_tokens=len(token_ids),
            object="input_tokens",
        )
    except Exception as e:
        logger.error(f"Local chat template failed: {e}")
        raise


def count_raw_tokens(
    input_text: str,
    model: str | None = None,
    base_url: str | None = None,
    use_server: bool = False,
    **kwargs,
) -> InputTokensResponse:
    """
    Count tokens using local tokenizer by default, or /v1/responses/input_tokens endpoint.
    """
    if model is None:
        model = LLM_MODEL
    if use_server:
        from .server_interaction import count_tokens_raw as server_count_tokens_raw

        return InputTokensResponse(
            input_tokens=server_count_tokens_raw(input_text, model, base_url, **kwargs),
            object="input_tokens",
        )
    tokenizer = get_tokenizer(model)
    return InputTokensResponse(
        input_tokens=len(tokenizer.encode(input_text)),
        object="input_tokens",
    )


def count_tokens_with_template(
    messages: List[Dict[str, str]],
    model: str | None = None,
    base_url: str | None = None,
    tools: List[Dict[str, Any]] | None = None,
    use_server: bool = False,
    **kwargs,
) -> int:
    """
    Convenience function that returns just the token count from chat endpoint.
    """
    result = count_chat_tokens(
        messages,
        model=model,
        base_url=base_url,
        tools=tools,
        use_server=use_server,
        **kwargs,
    )
    return result["input_tokens"]


def count_tokens_raw(
    input_text: str,
    model: str | None = None,
    base_url: str | None = None,
    use_server: bool = False,
    **kwargs,
) -> int:
    """
    Convenience function that returns just the token count from raw endpoint.
    """
    result = count_raw_tokens(
        input_text,
        model=model,
        base_url=base_url,
        use_server=use_server,
        **kwargs,
    )
    return result["input_tokens"]
