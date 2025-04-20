import json
from typing import Optional
from jet.transformers.object import make_serializable


def prettify_value(prompt, level=0):
    """
    Recursively builds a formatted log string from a nested dictionary or list.

    :param prompt: Dictionary or list to process.
    :param level: Indentation level for nested structures.
    :return: Formatted string for the log.
    """
    prompt_log = ""
    indent = " " * level  # Indentation for nested structures
    marker_list = ["-", "+"]
    marker = marker_list[level % 2]
    line_prefix = indent if level == 0 else f"{indent} {marker} "

    if isinstance(prompt, dict):
        for key, value in prompt.items():
            capitalized_key = key.capitalize()
            if isinstance(value, (dict, list)):  # If nested structure
                prompt_log += f"{line_prefix}{capitalized_key}:\n"
                prompt_log += prettify_value(value, level + 1)
            else:  # Primitive value
                prompt_log += f"{line_prefix}{capitalized_key}: {value}\n"
    elif isinstance(prompt, list):
        for item in prompt:
            if isinstance(item, (dict, list)):  # If nested structure
                prompt_log += prettify_value(item, level + 1)
            else:  # Primitive value
                prompt_log += f"{line_prefix}{item}\n"

    return prompt_log


def format_json(value, indent: Optional[int] = 2):
    serialized = make_serializable(value)
    return json.dumps(serialized, indent=indent)


# Example Usage
if __name__ == "__main__":
    prompt = {
        "user": "Alice",
        "attributes": {
            "age": 30,
            "preferences": ["running", "cycling", {"nested": "value"}],
            "contact": {
                "email": "alice@example.com",
                "phone": "123-456-7890"
            }
        },
        "status": "active"
    }
    prompt_log = prettify_value(prompt)
    print(prompt_log)
