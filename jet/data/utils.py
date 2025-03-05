import uuid
import json
from typing import Any


def generate_unique_hash() -> str:
    """Generate a unique UUID v4 string."""
    return str(uuid.uuid4())


def generate_key(*args: Any) -> str:
    """
    Generate a UUID v5 key based on the concatenation of input arguments.

    Args:
        *args: Variable length argument list.

    Returns:
        A UUID string in the standard 36-character format.
    """
    try:
        # Convert input arguments into a deterministic JSON string
        concatenated = json.dumps(args, separators=(',', ':'))
        # Generate a UUID v5 based on a fixed namespace and the input data
        key = uuid.uuid5(uuid.NAMESPACE_DNS, concatenated)
        return str(key)
    except TypeError as e:
        raise ValueError(f"Invalid argument provided: {e}")
