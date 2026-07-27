from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

import requests
from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.model_utils import get_llama_cpp_base_url, get_model_hf_id
from jet.adapters.llama_cpp.types import LLAMACPP_KEYS
from jet.logger import logger
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

# ---------------------------------------------------------------------------
# Tokenizer cache (mirrors jet/_token/token_utils.py pattern)
# ---------------------------------------------------------------------------
TOKENIZER_CACHE: dict[str, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = {}


def clear_tokenizer_cache() -> None:
    """Clear all cached tokenizers — useful in tests / REPL"""
    TOKENIZER_CACHE.clear()
    logger.debug("Tokenizer cache cleared")


def _normalize_model_key(model_name: str | None) -> str:
    """Create consistent cache key"""
    if model_name is None:
        return "__default__"
    return model_name.lower().strip()


# ---------------------------------------------------------------------------
# Core tokenizer getter
# ---------------------------------------------------------------------------
def get_tokenizer(
    model_name: Optional[str | LLAMACPP_KEYS] = None,
    cache: bool = True,
    verbose: bool = False,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """
    Get a HuggingFace tokenizer for a llama.cpp model key.

    Resolves the model key to a HuggingFace ID using `get_model_hf_id`,
    then loads the tokenizer via AutoTokenizer. Results are cached by default.

    Args:
        model_name: llama.cpp model key (e.g., "llama-3.2:3b") or HF ID directly.
                    Defaults to LLM_MODEL from config if None.
        cache: Whether to cache the tokenizer (default True).
        verbose: Log cache hits/misses when True.

    Returns:
        PreTrainedTokenizer or PreTrainedTokenizerFast

    Raises:
        ValueError: If model key cannot be resolved.
    """
    if model_name is None:
        model_name = LLM_MODEL
        logger.debug(f"No model specified, using default: {model_name}")

    key = _normalize_model_key(model_name)

    # Check cache
    if cache and key in TOKENIZER_CACHE:
        if verbose:
            logger.debug(f"[tokenizer cache] HIT → {key}")
        return TOKENIZER_CACHE[key]

    if verbose:
        logger.debug(f"[tokenizer cache] MISS → loading {key}")

    # Resolve model key to HuggingFace ID
    try:
        hf_id = get_model_hf_id(model_name)
        logger.debug(f"Resolved '{model_name}' → '{hf_id}'")
    except ValueError:
        # If not found in mapping, try using model_name directly as HF ID
        logger.debug(f"Model '{model_name}' not in mapping, trying as direct HF ID")
        hf_id = model_name

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        logger.debug(f"Successfully loaded tokenizer for '{hf_id}'")
    except Exception as e:
        logger.error(f"Failed to load tokenizer for '{hf_id}': {e}")
        raise ValueError(
            f"Cannot load tokenizer for model '{model_name}' (HF: {hf_id})"
        ) from e

    # Cache and return
    if cache:
        TOKENIZER_CACHE[key] = tokenizer

    return tokenizer


# ---------------------------------------------------------------------------
# Convenience functions (mirrors jet/_token/token_utils.py)
# ---------------------------------------------------------------------------
def get_tokenizer_fn(
    model_name: Optional[str | LLAMACPP_KEYS] = None,
    add_special_tokens: bool = False,
) -> Callable[[Union[str, list[str]]], Union[list[int], list[list[int]]]]:
    """
    Returns a callable that tokenizes text using the local tokenizer.

    Args:
        model_name: llama.cpp model key or HF ID.
        add_special_tokens: Whether to include special tokens.

    Returns:
        Callable that takes str or list[str] and returns token IDs.
    """
    tokenizer = get_tokenizer(model_name)

    def _fn(text: Union[str, list[str]]) -> Union[list[int], list[list[int]]]:
        if isinstance(text, str):
            return tokenizer.encode(text, add_special_tokens=add_special_tokens)
        else:
            return tokenizer.batch_encode_plus(
                text,
                add_special_tokens=add_special_tokens,
            )["input_ids"]

    return _fn


def get_detokenizer_fn(
    model_name: Optional[str | LLAMACPP_KEYS] = None,
    skip_special_tokens: bool = True,
) -> Callable[[Union[list[int], list[list[int]]]], Union[str, list[str]]]:
    """
    Returns a callable that detokenizes token IDs using the local tokenizer.

    Args:
        model_name: llama.cpp model key or HF ID.
        skip_special_tokens: Whether to remove special tokens.

    Returns:
        Callable that takes list[int] or list[list[int]] and returns text.
    """
    tokenizer = get_tokenizer(model_name)

    def _fn(tokens: Union[list[int], list[list[int]]]) -> Union[str, list[str]]:
        # Check if batch
        if tokens and isinstance(tokens[0], list):
            return tokenizer.batch_decode(
                tokens,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=True,
            )
        else:
            return tokenizer.decode(
                tokens,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=True,
            )

    return _fn


# ---------------------------------------------------------------------------
# Existing TypedDicts (unchanged)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Server functions (updated with use_server: bool = False)
# ---------------------------------------------------------------------------
def tokenize(
    content: str,
    add_special: bool = False,
    parse_special: bool = True,
    with_pieces: bool = False,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    use_server: bool = False,
) -> TokenizeResponse:
    """
    Tokenize text using local HuggingFace tokenizer by default, or llama.cpp /tokenize endpoint.

    Args:
        content: Text to tokenize.
        add_special: Add special tokens (BOS/EOS).
        parse_special: Parse special tokens in content (server only).
        with_pieces: Include token pieces in response.
        base_url: Server base URL override (server only).
        model: Model name (defaults to LLM_MODEL).
        use_server: If True, use llama.cpp server endpoint instead of local tokenizer.

    Returns:
        TokenizeResponse with tokens list.
    """
    if model is None:
        model = LLM_MODEL

    # Server path (opt-in)
    if use_server:
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

    # Local path (default)
    tokenizer = get_tokenizer(model)
    token_ids = tokenizer.encode(content, add_special_tokens=add_special)

    if with_pieces:
        # Decode each token to get pieces (approximate)
        tokens: List[Union[int, Token]] = []
        for tid in token_ids:
            piece = tokenizer.decode([tid])
            tokens.append(Token(id=tid, piece=piece))
        return TokenizeResponse(tokens=tokens)
    else:
        return TokenizeResponse(tokens=token_ids)


def detokenize(
    tokens: List[int],
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    use_server: bool = False,
    skip_special_tokens: bool = True,
) -> DetokenizeResponse:
    """
    Convert token IDs back to text using local tokenizer by default, or /detokenize endpoint.

    Args:
        tokens: List of token IDs.
        base_url: Server base URL override (server only).
        model: Model name (defaults to LLM_MODEL).
        use_server: If True, use llama.cpp server endpoint instead of local tokenizer.
        skip_special_tokens: Skip special tokens when decoding.

    Returns:
        DetokenizeResponse with content string.
    """
    if model is None:
        model = LLM_MODEL

    # Server path (opt-in)
    if use_server:
        url = f"{get_llama_cpp_base_url(override=base_url)}/detokenize"
        payload: Dict[str, Any] = {
            "model": model,
            "tokens": tokens,
        }
        logger.debug(f"Detokenizing via server: {url} (model={model})")
        response = requests.post(url, json=payload, timeout=30.0)
        response.raise_for_status()
        return response.json()

    # Local path (default)
    logger.debug(f"Using local detokenizer for model: {model}")
    tokenizer = get_tokenizer(model)
    content = tokenizer.decode(
        tokens,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=True,
    )
    return DetokenizeResponse(content=content)


def count_tokens(
    content: Union[str, List[Union[int, str, List[int], Dict[str, str]]]],
    add_special: bool = False,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    use_server: bool = False,
    **kwargs,
) -> int:
    """
    Count tokens using local tokenizer by default, with intelligent detection of input type.

    - If content is a string: direct tokenization
    - If content is a list of message dicts: applies chat template
    - If content is a list of tokens: returns len(content)

    Args:
        content: String, list of message dicts, or list of tokens.
        add_special: Whether to add special tokens (string input only).
        base_url: Base URL override (server only).
        model: Model name (uses LLM_MODEL if None).
        tools: Optional list of tool definitions (for message dicts).
        use_server: If True, use llama.cpp server endpoints instead of local tokenizer.
        **kwargs: Additional parameters to pass to the endpoint (server only).

    Returns:
        int: Number of tokens.
    """
    if model is None:
        model = LLM_MODEL

    # Server path (opt-in)
    if use_server or not isinstance(content, str):
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
                    use_server=True,
                )
                return len(result["tokens"])

        if isinstance(content, list):
            if content and all(
                isinstance(item, dict) and "role" in item and "content" in item
                for item in content
            ):
                try:
                    result = count_tokens_with_template(
                        content,
                        model=model,
                        base_url=base_url,
                        tools=tools,
                        use_server=True,
                        **kwargs,
                    )
                    return result
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
                        use_server=True,
                    )
                    return len(result["tokens"])
            else:
                return len(content)

        logger.warning(f"Unsupported content type: {type(content)}")
        return 0

    # Local path (default)
    tokenizer = get_tokenizer(model)

    if isinstance(content, str):
        return len(tokenizer.encode(content, add_special_tokens=add_special))

    if isinstance(content, list):
        if content and all(
            isinstance(item, dict) and "role" in item and "content" in item
            for item in content
        ):
            # Apply chat template for accurate count
            try:
                token_ids = tokenizer.apply_chat_template(
                    content,
                    tools=tools,
                    tokenize=True,
                    add_generation_prompt=False,
                )
                return len(token_ids)
            except Exception as e:
                logger.warning(
                    f"Chat template failed, falling back to concatenation: {e}"
                )
                combined = " ".join(msg.get("content", "") for msg in content)
                return len(tokenizer.encode(combined, add_special_tokens=add_special))
        else:
            # Token list
            return len(content)

    logger.warning(f"Unsupported content type: {type(content)}")
    return 0


# ---------------------------------------------------------------------------
# Remaining server functions (updated with use_server: bool = False)
# ---------------------------------------------------------------------------
def count_chat_tokens(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    use_server: bool = False,
    **kwargs,
) -> InputTokensResponse:
    """
    Count tokens using local tokenizer by default, or /v1/chat/completions/input_tokens endpoint.

    Applies the chat template (e.g., <|im_start|>, <|im_end|> for Qwen)
    and counts special tokens added by the template. Includes tool definition
    tokens in the count.

    Args:
        messages: List of message dicts with 'role' and 'content'.
        model: Model name (uses LLM_MODEL if None).
        base_url: Base URL override (server only).
        tools: Optional list of tool definitions.
        use_server: If True, use llama.cpp server endpoint instead of local tokenizer.
        **kwargs: Additional parameters to pass to the endpoint (server only).

    Returns:
        InputTokensResponse with input_tokens count.
    """
    if model is None:
        model = LLM_MODEL

    # Server path (opt-in)
    if use_server:
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

    # Local path (default)
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
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    use_server: bool = False,
    **kwargs,
) -> InputTokensResponse:
    """
    Count tokens using local tokenizer by default, or /v1/responses/input_tokens endpoint.

    Direct tokenization of raw text. No chat template applied.

    Args:
        input_text: Raw text to tokenize.
        model: Model name (uses LLM_MODEL if None).
        base_url: Base URL override (server only).
        use_server: If True, use llama.cpp server endpoint instead of local tokenizer.
        **kwargs: Additional parameters to pass to the endpoint (server only).

    Returns:
        InputTokensResponse with input_tokens count.
    """
    if model is None:
        model = LLM_MODEL

    # Server path (opt-in)
    if use_server:
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
        return result

    # Local path (default)
    tokenizer = get_tokenizer(model)
    return InputTokensResponse(
        input_tokens=len(tokenizer.encode(input_text)),
        object="input_tokens",
    )


def count_tokens_with_template(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    use_server: bool = False,
    **kwargs,
) -> int:
    """
    Convenience function that returns just the token count from chat endpoint.

    Args:
        messages: List of message dicts with 'role' and 'content'.
        model: Model name (uses LLM_MODEL if None).
        base_url: Base URL override (server only).
        tools: Optional list of tool definitions.
        use_server: If True, use llama.cpp server endpoint instead of local tokenizer.
        **kwargs: Additional parameters to pass to the endpoint (server only).

    Returns:
        int: Number of tokens.
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
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    use_server: bool = False,
    **kwargs,
) -> int:
    """
    Convenience function that returns just the token count from raw endpoint.

    Args:
        input_text: Raw text to tokenize.
        model: Model name (uses LLM_MODEL if None).
        base_url: Base URL override (server only).
        use_server: If True, use llama.cpp server endpoint instead of local tokenizer.
        **kwargs: Additional parameters to pass to the endpoint (server only).

    Returns:
        int: Number of tokens.
    """
    result = count_raw_tokens(
        input_text,
        model=model,
        base_url=base_url,
        use_server=use_server,
        **kwargs,
    )
    return result["input_tokens"]


# ---------------------------------------------------------------------------
# Main (updated for use_server opt-in)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test local tokenizer functions
    print("=== Testing get_tokenizer ===")
    try:
        tokenizer = get_tokenizer("llama-3.2:3b", verbose=True)
        print(f"Tokenizer loaded: {type(tokenizer).__name__}")
    except Exception as e:
        print(f"Local tokenizer not available: {e}")

    print("\n=== Testing get_tokenizer_fn ===")
    try:
        encode_fn = get_tokenizer_fn("llama-3.2:3b")
        result = encode_fn("Hello world!")
        print(f"Tokenized: {result[:5]}...")
    except Exception as e:
        print(f"get_tokenizer_fn failed: {e}")

    print("\n=== Testing get_detokenizer_fn ===")
    try:
        decode_fn = get_detokenizer_fn("llama-3.2:3b")
        result = decode_fn([123, 456, 789])
        print(f"Detokenized: {result}")
    except Exception as e:
        print(f"get_detokenizer_fn failed: {e}")

    # Local tokenization (default)
    print("\n=== Local tokenization (default) ===")
    try:
        local_tokens = tokenize("Hello world!", with_pieces=True)
        print("Local tokens:", local_tokens["tokens"][:5], "...")

        local_text = detokenize([123, 456, 789])
        print("Local detokenized:", local_text["content"])

        local_count = count_tokens("This is a test prompt.")
        print("Local token count:", local_count)
    except Exception as e:
        print(f"Local operations failed: {e}")

    # Server tokenization (opt-in)
    print("\n=== Server tokenization (use_server=True) ===")
    try:
        server_tokens = tokenize(
            "Hello world!", add_special=True, with_pieces=True, use_server=True
        )
        print("Server tokens:", server_tokens["tokens"][:5], "...")

        server_text = detokenize([123, 456, 789], use_server=True)
        print("Server detokenized:", server_text["content"])

        server_count = count_tokens("This is a test prompt.", use_server=True)
        print("Server token count:", server_count)
    except Exception as e:
        print(f"Server operations failed (is server running?): {e}")

    # Message-based tests
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    print("\n=== Local chat tokens (default) ===")
    try:
        chat_result_local = count_chat_tokens(messages)
        print(f"Local chat tokens: {chat_result_local['input_tokens']}")
    except Exception as e:
        print(f"Local chat count failed: {e}")

    print("\n=== Server chat tokens (use_server=True) ===")
    try:
        chat_result_server = count_chat_tokens(messages, use_server=True)
        print(f"Server chat tokens: {chat_result_server['input_tokens']}")
    except Exception as e:
        print(f"Server chat count failed: {e}")

    # Auto-detection tests
    print("\n=== Testing count_tokens with auto-detection ===")
    result = count_tokens("Hello, how are you?")
    print(f"String input (local): {result} tokens")

    result = count_tokens("Hello, how are you?", use_server=True)
    print(f"String input (server): {result} tokens")

    result = count_tokens(messages)
    print(f"Message dicts (local): {result} tokens")

    result = count_tokens([123, 456, 789])
    print(f"Token list input: {result} tokens")
