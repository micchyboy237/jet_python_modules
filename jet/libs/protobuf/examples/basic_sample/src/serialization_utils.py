"""Utility functions for protobuf serialization."""
from pathlib import Path
from typing import TypeVar
import google.protobuf.message

T = TypeVar('T', bound=google.protobuf.message.Message)

def save_proto_to_file(
    message: T, 
    filepath: Path | str, 
    deterministic: bool = True
) -> None:
    """Save protobuf message to binary file."""
    serialized = message.SerializeToString(deterministic=deterministic)
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(serialized)

def load_proto_from_file(
    message_type: type[T], 
    filepath: Path | str
) -> T:
    """Load protobuf message from binary file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    message = message_type()
    with open(filepath, "rb") as f:
        message.ParseFromString(f.read())
    return message

def proto_to_json(message: google.protobuf.message.Message) -> str:
    """Convert protobuf to JSON string (requires google.protobuf.json_format)."""
    try:
        from google.protobuf.json_format import MessageToJson
        return MessageToJson(message, including_default_value_fields=True)
    except ImportError:
        raise ImportError("Install protobuf[json] for JSON conversion")