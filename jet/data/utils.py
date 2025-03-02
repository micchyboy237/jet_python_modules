import hashlib
import json
from typing import Any
import uuid


def generate_unique_hash(hash_length=24):
    # Generate a unique UUID and convert it to a string
    unique_hash = str(uuid.uuid4()).replace('-', '')  # Remove dashes
    # Return the hash truncated to the specified length
    return unique_hash[:hash_length]


def generate_key(*args: Any) -> str:
    """
    Generate a SHA256 hash key from the concatenation of input arguments.

    Args:
        *args: Variable length argument list.

    Returns:
        A SHA256 hash string.
    """
    try:
        # Combine the arguments into a JSON string
        concatenated = json.dumps(args, separators=(',', ':'))
        # Generate a SHA256 hash of the concatenated string
        key = hashlib.sha256(concatenated.encode()).hexdigest()
        return key
    except TypeError as e:
        raise ValueError(f"Invalid argument provided: {e}")
