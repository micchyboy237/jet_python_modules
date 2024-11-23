import json


def make_serializable(obj):
    """
    Recursively converts an object's attributes to be serializable.

    Args:
        obj: The input object to process.

    Returns:
        A serializable dictionary representation of the object.
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
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
    else:
        return str(obj)  # Fallback for non-serializable objects


# Example usage
if __name__ == "__main__":
    class CustomObject:
        def __init__(self, name, values):
            self.name = name
            self.values = values
            self.metadata = {"key": "value", "nested": [1, 2, 3]}

    obj = CustomObject("Test", [1, 2, {"a": "b"}])
    serializable_obj = make_serializable(obj)

    # Serialize to JSON for testing
    json_data = json.dumps(serializable_obj, indent=2)
    print(json_data)
