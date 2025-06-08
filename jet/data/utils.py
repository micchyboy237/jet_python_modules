import hashlib
import uuid
import json
from typing import Any


def generate_unique_hash() -> str:
    """Generate a unique UUID v4 string."""
    return str(uuid.uuid4())


def generate_key(*args: Any, **kwargs: Any) -> str:
    """
    Generate a UUID v1 key that sorts as the latest, validating input arguments.

    Args:
        *args: Variable length argument list.
        **kwargs: Variable length keyword arguments.

    Returns:
        A UUID string in the standard 36-character format that sorts as the latest.

    Raises:
        ValueError: If any provided argument cannot be serialized to JSON.
    """
    try:
        # Validate that inputs can be serialized to JSON
        input_data = {"args": args, "kwargs": kwargs}
        json.dumps(input_data, sort_keys=True, separators=(',', ':'))
        # Generate a UUID v1 based on the host ID, sequence number, and current time
        key = uuid.uuid1()
        return str(key)
    except TypeError as e:
        raise ValueError(f"Invalid argument provided: {e}")


def hash_text(text: str | list[str]) -> str:
    """Generate a unique hash for a given text input."""
    return hashlib.sha256(json.dumps(text, sort_keys=True).encode()).hexdigest()
