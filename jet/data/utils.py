import hashlib
import uuid
import json
from typing import Any


def generate_unique_hash() -> str:
    """Generate a unique UUID v4 string."""
    return str(uuid.uuid4())


def generate_key(*args: Any, **kwargs: Any) -> str:
    """
    Generate a UUID v5 key based on the concatenation of input arguments and keyword arguments.

    Args:
        *args: Variable length argument list.
        **kwargs: Variable length keyword arguments.

    Returns:
        A UUID string in the standard 36-character format.

    Raises:
        ValueError: If any provided argument cannot be serialized to JSON.
    """
    try:
        # Combine args and kwargs into a deterministic JSON string
        input_data = {"args": args, "kwargs": kwargs}
        concatenated = json.dumps(
            input_data, sort_keys=True, separators=(',', ':'))
        # Generate a UUID v5 based on a fixed namespace and the input data
        key = uuid.uuid5(uuid.NAMESPACE_DNS, concatenated)
        return str(key)
    except TypeError as e:
        raise ValueError(f"Invalid argument provided: {e}")


def hash_text(text: str | list[str]) -> str:
    """Generate a unique hash for a given text input."""
    return hashlib.sha256(json.dumps(text, sort_keys=True).encode()).hexdigest()
