from typing import Any, Dict, List, Optional, TypedDict, Union

import requests
from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.model_utils import get_llama_cpp_base_url
from jet.logger import logger


class Token(TypedDict):
    id: int
    piece: Union[str, List[int]]


class TokenizeResponse(TypedDict):
    tokens: List[Union[int, Token]]


class DetokenizeResponse(TypedDict):
    content: str


class InputTokensResponse(TypedDict):
    input_tokens: int
    object: str


def tokenize(
    content: str,
    add_special: bool = False,
    parse_special: bool = True,
    with_pieces: bool = False,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> TokenizeResponse:
    """
    Tokenize text via llama.cpp /tokenize endpoint.
    Note: This is a NATIVE llama.cpp endpoint (no /v1 prefix).
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
    logger.debug(f"Tokenizing: {content[:50]}...")
    response = requests.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()


def detokenize(
    tokens: List[int],
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> DetokenizeResponse:
    """
    Convert token IDs back to text via /detokenize.
    """
    if model is None:
        model = LLM_MODEL

    url = f"{get_llama_cpp_base_url(override=base_url)}/detokenize"
    payload: Dict[str, Any] = {
        "model": model,
        "tokens": tokens,
    }
    logger.debug(f"Detokenizing {len(tokens)} tokens...")
    response = requests.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()


def count_tokens(
    content: Union[str, List[Union[int, str, List[int], Dict[str, str]]]],
    add_special: bool = False,  # Only used for string input
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> int:
    """
    Count tokens in content with intelligent detection of input type.

    - If content is a string: uses count_tokens_raw() (no template)
    - If content is a list of message dicts: uses count_tokens_with_template() (with template)
    - If content is a list of tokens: returns len(content)

    Args:
        content: String, list of message dicts, or list of tokens
        add_special: Whether to add special tokens (string input only)
        base_url: Base URL override
        model: Model name (uses LLM_MODEL if None)
        tools: Optional list of tool definitions (for message dicts)
        **kwargs: Additional parameters to pass to the endpoint

    Returns:
        int: Number of tokens

    Examples:
        # String input (raw tokenization)
        count_tokens("Hello, how are you?")

        # Message dicts (with template)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        count_tokens(messages)

        # Message dicts with tools
        tools = [{"type": "function", "function": {"name": "get_weather", ...}}]
        count_tokens(messages, tools=tools)

        # Token list (direct count)
        count_tokens([123, 456, 789])
    """
    # Case 1: String input → use raw token counting
    if isinstance(content, str):
        try:
            result = count_tokens_raw(content, model=model, base_url=base_url, **kwargs)
            return result
        except Exception as e:
            logger.warning(f"Raw token count failed, falling back to tokenize: {e}")
            # Fallback to legacy tokenize if new endpoint fails
            result = tokenize(
                content,
                add_special=add_special,
                base_url=base_url,
                model=model,
            )
            return len(result["tokens"])

    # Case 2: List input
    if isinstance(content, list):
        # Check if it's a list of message dicts
        if content and all(
            isinstance(item, dict) and "role" in item and "content" in item
            for item in content
        ):
            try:
                result = count_tokens_with_template(
                    content,  # messages
                    model=model,
                    base_url=base_url,
                    tools=tools,
                    **kwargs,
                )
                return result
            except Exception as e:
                logger.warning(
                    f"Chat token count failed, falling back to tokenize: {e}"
                )
                # Fallback: concatenate and tokenize
                combined = " ".join(msg.get("content", "") for msg in content)
                result = tokenize(
                    combined,
                    add_special=add_special,
                    base_url=base_url,
                    model=model,
                )
                return len(result["tokens"])
        else:
            # Assume it's a list of tokens
            return len(content)

    # Case 3: Unknown type
    logger.warning(f"Unsupported content type: {type(content)}")
    return 0


# ============================================================================
# NEW: Token counting endpoints (from llama_cpp_token_counting.md)
# ============================================================================


def count_chat_tokens(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> InputTokensResponse:
    """
    Count tokens using /v1/chat/completions/input_tokens endpoint.

    This applies the chat template (e.g., <|im_start|>, <|im_end|> for Qwen)
    and counts special tokens added by the template. Includes tool definition
    tokens in the count. More accurate for actual generation scenarios.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name (uses LLM_MODEL if None)
        base_url: Base URL override
        tools: Optional list of tool definitions
        **kwargs: Additional parameters to pass to the endpoint

    Returns:
        InputTokensResponse with input_tokens count

    Example:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        result = count_chat_tokens(messages)
        print(f"Tokens: {result['input_tokens']}")
    """
    if model is None:
        model = LLM_MODEL

    base = get_llama_cpp_base_url(override=base_url)
    url = f"{base}/v1/chat/completions/input_tokens"

    payload: Dict[str, Any] = {"model": model, "messages": messages, **kwargs}

    if tools:
        payload["tools"] = tools

    logger.debug(
        f"Counting chat tokens with {len(messages)} messages, {len(tools or [])} tools"
    )
    response = requests.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    result = response.json()
    logger.info(f"Chat token count response: {result['input_tokens']} input tokens")
    return result


def count_raw_tokens(
    input_text: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> InputTokensResponse:
    """
    Count tokens using /v1/responses/input_tokens endpoint.

    Direct tokenization of raw text. No chat template applied.
    Faster for simple cases. May give different counts than chat endpoint
    for the same text.

    Args:
        input_text: Raw text to tokenize
        model: Model name (uses LLM_MODEL if None)
        base_url: Base URL override
        **kwargs: Additional parameters to pass to the endpoint

    Returns:
        InputTokensResponse with input_tokens count

    Example:
        result = count_raw_tokens("Hello, how are you?")
        print(f"Tokens: {result['input_tokens']}")
    """
    if model is None:
        model = LLM_MODEL

    base = get_llama_cpp_base_url(override=base_url)
    url = f"{base}/v1/responses/input_tokens"

    payload: Dict[str, Any] = {"model": model, "input": input_text, **kwargs}

    logger.debug(f"Counting raw tokens for: {input_text[:50]}...")
    response = requests.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    result = response.json()
    logger.info(f"Raw token count response: {result['input_tokens']} input tokens")
    return result


def count_tokens_with_template(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> int:
    """
    Convenience function that returns just the token count from chat endpoint.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name (uses LLM_MODEL if None)
        base_url: Base URL override
        tools: Optional list of tool definitions
        **kwargs: Additional parameters to pass to the endpoint

    Returns:
        int: Number of tokens
    """
    result = count_chat_tokens(messages, model, base_url, tools, **kwargs)
    return result["input_tokens"]


def count_tokens_raw(
    input_text: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> int:
    """
    Convenience function that returns just the token count from raw endpoint.

    Args:
        input_text: Raw text to tokenize
        model: Model name (uses LLM_MODEL if None)
        base_url: Base URL override
        **kwargs: Additional parameters to pass to the endpoint

    Returns:
        int: Number of tokens
    """
    result = count_raw_tokens(input_text, model, base_url, **kwargs)
    return result["input_tokens"]


if __name__ == "__main__":
    # Test 1: Legacy tokenize
    tokens_resp = tokenize("Hello world!", add_special=True, with_pieces=True)
    print("Tokens:", tokens_resp["tokens"][:5], "...")

    # Test 2: Legacy detokenize
    text_resp = detokenize([123, 456, 789])
    print("Detokenized:", text_resp["content"])

    # Test 3: Legacy count
    num_tokens = count_tokens("This is a test prompt.")
    print("Legacy token count:", num_tokens)

    # Test 4: New chat token counting (with template)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]
    try:
        chat_result = count_chat_tokens(messages)
        print(f"Chat tokens (with template): {chat_result['input_tokens']}")
    except Exception as e:
        print(f"Chat token count failed: {e}")

    # Test 5: New raw token counting (no template)
    try:
        raw_result = count_raw_tokens("Hello, how are you?")
        print(f"Raw tokens (no template): {raw_result['input_tokens']}")
    except Exception as e:
        print(f"Raw token count failed: {e}")

    # Test 6: Convenience functions
    try:
        tokens_with_template = count_tokens_with_template(messages)
        print(f"Tokens with template (convenience): {tokens_with_template}")
    except Exception as e:
        print(f"Convenience chat count failed: {e}")

    try:
        tokens_raw = count_tokens_raw("Hello, how are you?")
        print(f"Tokens raw (convenience): {tokens_raw}")
    except Exception as e:
        print(f"Convenience raw count failed: {e}")

    # Test 7: With tools (if available)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]
    try:
        chat_with_tools = count_chat_tokens(messages, tools=tools)
        print(f"Chat tokens with tools: {chat_with_tools['input_tokens']}")
    except Exception as e:
        print(f"Chat with tools count failed: {e}")

    # Test 8: Updated count_tokens with auto-detection
    print("\n=== Testing count_tokens with auto-detection ===")

    # String input
    result = count_tokens("Hello, how are you?")
    print(f"String input: {result} tokens")

    # Message dicts input
    result = count_tokens(messages)
    print(f"Message dicts input: {result} tokens")

    # Message dicts with tools
    result = count_tokens(messages, tools=tools)
    print(f"Message dicts with tools: {result} tokens")

    # Token list input
    result = count_tokens([123, 456, 789])
    print(f"Token list input: {result} tokens")
