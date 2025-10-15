import json
import base64
import numpy as np
import types
import collections.abc
from typing import Any, Dict
from enum import Enum
from jet.transformers.text import to_snake_case
from jet.utils.class_utils import get_non_empty_attributes
from pydantic.main import BaseModel
from dataclasses import is_dataclass, asdict

def convert_dict_keys_to_snake_case(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert dictionary keys to snake_case."""
    if not isinstance(d, dict):
        return d
    return {
        to_snake_case(k): [convert_dict_keys_to_snake_case(item) for item in v]
        if isinstance(v, list) else convert_dict_keys_to_snake_case(v)
        if isinstance(v, dict) else v
        for k, v in d.items()
    }

def make_serializable(obj, seen=None):
    """
    Recursively converts an object's attributes to be serializable.
    Args:
        obj: The input object to process.
        seen: Set of object IDs to track processed objects and prevent circular references.
    Returns:
        A serializable representation of the object.
    """
    if seen is None:
        seen = set()

    def _serialize_inner(inner_obj, seen):
        # Handle Protobuf objects from google._upb._message
        if hasattr(inner_obj, '__class__') and inner_obj.__class__.__module__.startswith('google._upb._message'):
            # Check if the object is iterable (but not a string or bytes)
            try:
                is_iterable = isinstance(inner_obj, collections.abc.Iterable) and not isinstance(inner_obj, (str, bytes))
            except TypeError:
                is_iterable = False
            if is_iterable:
                try:
                    # Serialize each item and apply snake_case to dictionaries
                    serialized_items = []
                    for item in inner_obj:
                        serialized_item = _serialize_inner(item, seen.copy())
                        if isinstance(serialized_item, dict):
                            serialized_item = convert_dict_keys_to_snake_case(serialized_item)
                        serialized_items.append(serialized_item)
                    return serialized_items
                except Exception:
                    return inner_obj.__class__.__name__  # Fallback to class name if iteration fails
            return inner_obj.__class__.__name__  # Non-iterable Protobuf objects

        if isinstance(inner_obj, Enum):
            return inner_obj.value
        elif isinstance(inner_obj, bytes):
            try:
                decoded_str = inner_obj.decode('utf-8')
            except UnicodeDecodeError:
                decoded_str = base64.b64encode(inner_obj).decode('utf-8')
            return _serialize_inner(decoded_str, seen.copy())
        elif isinstance(inner_obj, (int, float, bool, type(None))):
            return inner_obj
        elif isinstance(inner_obj, str):
            try:
                parsed_obj = json.loads(inner_obj)
                if isinstance(parsed_obj, (dict, list)):
                    return _serialize_inner(parsed_obj, seen.copy())
                return inner_obj
            except json.JSONDecodeError:
                return inner_obj
        elif isinstance(inner_obj, (types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
            return str(type(inner_obj))
        elif isinstance(inner_obj, (np.integer, np.floating)):
            return inner_obj.item()
        elif isinstance(inner_obj, np.ndarray):
            return inner_obj.tolist()
        elif isinstance(inner_obj, BaseModel):
            try:
                return _serialize_inner(inner_obj.model_dump(), seen.copy())
            except (AttributeError, TypeError):
                return _serialize_inner(vars(inner_obj), seen.copy())
        elif is_dataclass(inner_obj):
            try:
                return _serialize_inner(asdict(inner_obj), seen.copy())
            except Exception:
                return _serialize_inner(vars(inner_obj), seen.copy())
        if id(inner_obj) in seen:
            return f"<{inner_obj.__class__.__name__} object>"
        new_seen = seen.copy()
        new_seen.add(id(inner_obj))
        if isinstance(inner_obj, set):
            return [_serialize_inner(item, new_seen) for item in inner_obj]
        elif isinstance(inner_obj, tuple):
            return [_serialize_inner(item, new_seen) for item in inner_obj]
        elif isinstance(inner_obj, list):
            return [_serialize_inner(item, new_seen) for item in inner_obj]
        elif isinstance(inner_obj, dict):
            serialized_dict = {}
            for key, value in inner_obj.items():
                serialized_key = str(key) if not isinstance(key, str) else key
                serialized_dict[serialized_key] = _serialize_inner(value, new_seen)
            return convert_dict_keys_to_snake_case(serialized_dict)
        elif hasattr(inner_obj, "__dict__"):
            try:
                if callable(getattr(inner_obj, "__dict__", None)):
                    dict_data = inner_obj.__dict__()
                else:
                    dict_data = get_non_empty_attributes(inner_obj)
                return _serialize_inner(dict_data, new_seen)
            except Exception:
                return str(inner_obj)
        else:
            return str(inner_obj)

    result = _serialize_inner(obj, seen)
    return convert_dict_keys_to_snake_case(result)

# Example usage
if __name__ == "__main__":
    byte_val = b'{"model": "llama3.2:latest"}'
    dict_bytes_val = {
        "key":  byte_val,
        "nested_bytes": {
            "nested_key":  byte_val
        }
    }
    obj = {
        "list": [4, 2, 3, 2, 5],
        "list_bytes": [byte_val, dict_bytes_val],
        "tuple": (4, 2, 3, 2, 5),
        "tuple_bytes": (byte_val, dict_bytes_val),
        "set": {4, 2, 3, 2, 5},
        "set_bytes": {byte_val, byte_val},
        "dict_bytes": dict_bytes_val
    }
    serializable_obj = make_serializable(obj)

    # Serialize to JSON for testing
    json_data = json.dumps(serializable_obj, indent=2)
    print(json_data)
