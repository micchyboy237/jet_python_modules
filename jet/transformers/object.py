from enum import Enum
import json
import base64
from typing import Any, Dict
from jet.transformers.text import to_snake_case
from jet.utils.class_utils import get_non_empty_attributes, is_class_instance
import numpy as np
import json
import base64
import numpy as np
from pydantic.main import BaseModel
# from jet.validation.object import is_iterable_but_not_primitive
from jet.logger import logger


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


def make_serializable(obj):
    """
    Recursively converts an object's attributes to be serializable.
    Args:
        obj: The input object to process.
    Returns:
        A serializable representation of the object.
    """
    if isinstance(obj, Enum):
        return obj.value  # Convert Enum to its value
    elif isinstance(obj, bytes):
        try:
            decoded_str = obj.decode('utf-8')
        except UnicodeDecodeError:
            decoded_str = base64.b64encode(obj).decode('utf-8')
        return make_serializable(decoded_str)
    elif isinstance(obj, (int, float, bool, type(None))):
        return obj
    elif isinstance(obj, str):
        try:
            parsed_obj = json.loads(obj)
            if isinstance(parsed_obj, (dict, list)):  # Only parse JSON objects or arrays
                return parsed_obj
            return obj  # Keep as string if it's a valid number or boolean
        except json.JSONDecodeError:
            return obj
    elif isinstance(obj, set):
        return make_serializable(list(obj))
    elif isinstance(obj, tuple):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        serialized_dict = {}
        for key, value in obj.items():
            serialized_key = str(key) if not isinstance(
                key, str) else key  # Ensure keys are strings
            serialized_dict[serialized_key] = make_serializable(
                value)  # Properly process values
        return serialized_dict
    elif isinstance(obj, BaseModel):
        try:
            return make_serializable(obj.model_dump())
        except (AttributeError, TypeError) as e:
            logger.warning(e)
            return make_serializable(vars(obj))
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Convert numpy types to native Python types
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif hasattr(obj, "__dict__"):  # Check this only after primitive and known types
        return make_serializable(get_non_empty_attributes(obj))
    else:
        return str(obj)  # Fallback for unsupported types


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
