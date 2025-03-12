import json
from typing import Any, Union
from jet.utils.class_utils import get_class_name


def check_object_type(obj: object, target_type: str) -> bool:
    if get_class_name(obj) == target_type:
        return True
    else:
        return False


def print_types_recursive(obj):
    if isinstance(obj, list):
        for item in obj:
            print_types_recursive(item)
    else:
        print(type(obj))


def get_values_by_paths(data: dict[str, Any], attr_paths: list[str]) -> list[Any]:
    values = []

    for path in attr_paths:
        keys = path.split('.')
        value = data

        # Traverse the dictionary based on the keys
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                value = None
                break

        values.append(value)

    return values


def extract_values_by_paths(data: dict[str, Any], attr_paths: list[str], is_flattened: bool = False) -> dict[str, Any]:
    result = {}

    for path in attr_paths:
        keys = path.split('.')
        value = data

        for i, key in enumerate(keys):
            if isinstance(value, dict) and key in value:
                value = value[key]

                if i == len(keys) - 1:
                    last_key = keys[-1]
                    if is_flattened:
                        if last_key in result:
                            raise ValueError(
                                f"Duplicate key conflict in flattened result: {last_key}")
                        result[last_key] = value
                    else:
                        current_dict = result
                        for k in keys[:-1]:
                            if k not in current_dict:
                                current_dict[k] = {}
                            current_dict = current_dict[k]
                        current_dict[last_key] = value
            else:
                break

    return result


def extract_null_keys(data: Union[dict, list], parent_key: str = "") -> list[str]:
    null_keys = []

    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            if value is None:
                null_keys.append(full_key)
            else:
                null_keys.extend(extract_null_keys(value, full_key))

    elif isinstance(data, list):
        for index, item in enumerate(data):
            full_key = f"{parent_key}"
            if item is None:
                null_keys.append(full_key)
            else:
                null_keys.extend(extract_null_keys(item, full_key))

    return null_keys


__all__ = [
    "check_object_type",
    "print_types_recursive",
    "get_values_by_paths",
    "extract_values_by_paths",
    "extract_null_keys",
]

# Example usage
if __name__ == "__main__":
    attr_paths = ['name', 'age', 'address.country']
    data = {
        "name": None,
        "age": "Thirty",
        "address": {
            "country": 0,
            "state": None
        }
    }

    result = get_values_by_paths(data, attr_paths)
    print(result)  # Output: [None, 'Thirty', 0]

    result = extract_values_by_paths(data, attr_paths)
    # Output: {'name': None, 'age': 'Thirty', 'address': {'country': 0}}
    print(result)

    result = extract_null_keys(data)
    # Output: ["name", "address.state"]
    print(result)

    # Check if an integer is of type 'int'
    number = 42
    print(check_object_type(number, 'int'))  # Output: True

    # Check if a string is of type 'str'
    text = "Hello, World!"
    print(check_object_type(text, 'str'))  # Output: True

    # Check if a list is of type 'list'
    items = [1, 2, 3]
    print(check_object_type(items, 'list'))  # Output: True

    regular_dict = {"name": "Alice", "age": 25}
    print(check_object_type(regular_dict, 'dict'))  # Output: True

    from dataclasses import dataclass

    class PersonTypedDict(dict):
        name: str
        age: int

    @dataclass
    class PersonDataClass:
        name: str
        age: int

    class PersonBlueprint:
        name: str
        age: int

    class PersonWithInit:
        def __init__(self, name: str, age: int):
            self.name = name
            self.age = age

    # Create instances
    person_typed_dict = PersonTypedDict(name="Alice", age=25)
    person_dataclass = PersonDataClass(name="John", age=30)
    person_typed_dict = PersonTypedDict(name="Alice", age=25)
    person_blueprint = PersonBlueprint()
    person_with_init = PersonWithInit(name="Bob", age=40)

    # Check each object type
    print(check_object_type(person_typed_dict, 'PersonTypedDict'))  # Output: True
    print(check_object_type(person_dataclass, 'PersonDataClass'))  # Output: True
    print(check_object_type(person_blueprint, 'PersonBlueprint'))  # Output: True
    print(check_object_type(person_with_init, 'PersonWithInit'))  # Output: True

    # Get exception object class name
    import traceback
    from jet.logger import logger
    try:
        json.loads("invalid_json")
    except Exception as e:
        logger.error(f"Error class name: {get_class_name(e)}")
        traceback.print_exc()
