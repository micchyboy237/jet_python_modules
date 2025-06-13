import json
import uuid
import hashlib
from typing import Any


def generate_unique_hash() -> str:
    """Generate a unique UUID v4 string."""
    return str(uuid.uuid4())


def generate_key(*args: Any, **kwargs: Any) -> str:
    """
    Generate a deterministic UUID v5 key based on input arguments.

    Args:
        *args: Variable length argument list.
        **kwargs: Variable length keyword arguments.

    Returns:
        A UUID v5 string in the standard 36-character format, derived from input data.

    Raises:
        ValueError: If any provided argument cannot be serialized to JSON.
    """
    try:
        # Serialize inputs to JSON for deterministic hashing
        input_data = {"args": args, "kwargs": kwargs}
        serialized_data = json.dumps(
            input_data, sort_keys=True, separators=(',', ':'))

        # Generate UUID v5 using a namespace and the serialized input
        namespace = uuid.NAMESPACE_DNS  # Standard namespace for consistency
        key = uuid.uuid5(namespace, serialized_data)
        return str(key)
    except TypeError as e:
        raise ValueError(f"Invalid argument provided: {e}")


def hash_text(text: str | list[str]) -> str:
    """Generate a unique hash for a given text input."""
    return hashlib.sha256(json.dumps(text, sort_keys=True).encode()).hexdigest()
