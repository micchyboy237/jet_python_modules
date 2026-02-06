import json

from jet.transformers.object import make_serializable


def format_json(value, indent: int | None = 2):
    serialized = make_serializable(value)
    return json.dumps(serialized, indent=indent)


# Example Usage
if __name__ == "__main__":
    # Example for prettify_value
    prompt = {
        "user": "Alice",
        "attributes": {
            "age": 30,
            "preferences": ["running", "cycling", {"nested": "value"}],
            "contact": {"email": "alice@example.com", "phone": "123-456-7890"},
        },
        "status": "active",
    }

    # Example for format_json
    json_output = format_json(prompt, indent=4)
    print("\nFormatted JSON:")
    print(json_output)
