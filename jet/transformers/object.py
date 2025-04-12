from enum import Enum
import json
import base64
from jet.utils.class_utils import get_non_empty_attributes, is_class_instance
import numpy as np
import json
import base64
import numpy as np
from pydantic.main import BaseModel
# from jet.validation.object import is_iterable_but_not_primitive
from jet.logger import logger


def make_serializable(obj, _seen=None):
    if _seen is None:
        _seen = set()

    def track(obj):
        return isinstance(obj, (dict, list, set, tuple, BaseModel, np.ndarray)) or hasattr(obj, '__dict__')

    if track(obj):
        obj_id = id(obj)
        if obj_id in _seen:
            return f"<circular_ref:{type(obj).__name__}>"
        _seen.add(obj_id)

    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')
        except UnicodeDecodeError:
            return base64.b64encode(obj).decode('utf-8')
    elif isinstance(obj, (int, float, bool, type(None))):
        return obj
    elif isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, (dict, list)):
                return make_serializable(parsed, _seen)
            return obj
        except json.JSONDecodeError:
            return obj
    elif isinstance(obj, set):
        return [make_serializable(item, _seen) for item in obj]
    elif isinstance(obj, tuple):
        return [make_serializable(item, _seen) for item in obj]
    elif isinstance(obj, list):
        return [make_serializable(item, _seen) for item in obj]
    elif isinstance(obj, dict):
        return {
            str(key): make_serializable(value, _seen)
            for key, value in obj.items()
        }
    elif isinstance(obj, BaseModel):
        try:
            return make_serializable(obj.model_dump(), _seen)
        except Exception:
            return make_serializable(vars(obj), _seen)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "__dict__"):
        return make_serializable(vars(obj), _seen)
    else:
        return str(obj)


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
