import json
import base64
import numpy as np


def make_serializable(obj):
    """
    Recursively converts an object's attributes to be serializable.
    Args:
        obj: The input object to process.
    Returns:
        A serializable representation of the object.
    """
    if isinstance(obj, (int, float, bool, type(None))):
        return obj
    elif isinstance(obj, str):
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            return obj
    elif isinstance(obj, bytes):
        try:
            decoded_str = obj.decode('utf-8')
        except UnicodeDecodeError:
            decoded_str = base64.b64encode(obj).decode('utf-8')
        return make_serializable(decoded_str)
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {make_serializable(key): make_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, "__dict__"):
        return make_serializable(vars(obj))
    elif isinstance(obj, tuple):
        return tuple(make_serializable(item) for item in obj)
    elif isinstance(obj, set):
        return {make_serializable(item) for item in obj}
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Convert numpy types to native Python types
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    else:
        return str(obj)  # Fallback for unsupported types


# Example usage
if __name__ == "__main__":
    obj = [
        1,
        "string",
        b'bytes1',
        {
            "a": "b",
            "c": b'\x00\x01\x02\x03',
            "nested": {
                "d":  b'{"model": "llama3.2:latest"}'
            }
        }
    ]
    serializable_obj = make_serializable(obj)

    # Serialize to JSON for testing
    json_data = json.dumps(serializable_obj, indent=2)
    print(json_data)
