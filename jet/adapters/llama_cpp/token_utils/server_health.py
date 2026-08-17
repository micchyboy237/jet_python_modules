from typing import Optional

import requests
from jet.adapters.llama_cpp.model_utils import get_llama_cpp_base_url

# Cache the server status to avoid repeated checks
_SERVER_AVAILABLE_CACHE = {"available": None, "last_check": None}


def is_server_available(
    base_url: Optional[str] = None, timeout: float = 2.0, use_cache: bool = True
) -> bool:
    """
    Check if the llama.cpp server is live and responding.
    Uses caching to avoid repeated checks in quick succession.
    """
    # Check cache if enabled
    if use_cache and _SERVER_AVAILABLE_CACHE["available"] is not None:
        return _SERVER_AVAILABLE_CACHE["available"]

    url = get_llama_cpp_base_url(override=base_url)

    # Try the health endpoint first (fastest)
    try:
        response = requests.get(f"{url}/health", timeout=timeout)
        if response.status_code == 200:
            _SERVER_AVAILABLE_CACHE["available"] = True
            return True
    except:
        pass

    # Fallback: Try OPTIONS on tokenize endpoint
    try:
        response = requests.options(f"{url}/tokenize", timeout=timeout)
        if response.status_code < 500:
            _SERVER_AVAILABLE_CACHE["available"] = True
            return True
    except:
        pass

    # Final fallback: Try a lightweight POST to tokenize
    try:
        response = requests.post(
            f"{url}/tokenize",
            json={
                "content": "ping",
                "model": "llama-3.2:3b",  # Use your default model
                "add_special": False,
                "parse_special": True,
                "with_pieces": False,
            },
            timeout=timeout,
        )
        if response.status_code == 200:
            _SERVER_AVAILABLE_CACHE["available"] = True
            return True
    except:
        pass

    _SERVER_AVAILABLE_CACHE["available"] = False
    return False


def clear_server_cache():
    """Reset the server availability cache"""
    _SERVER_AVAILABLE_CACHE["available"] = None
    _SERVER_AVAILABLE_CACHE["last_check"] = None
