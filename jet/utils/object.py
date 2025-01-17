import json
from typing import TypedDict
from dataclasses import is_dataclass
from types import MappingProxyType
from jet.utils.class_utils import get_class_name


def check_object_type(obj: object, target_type: str) -> bool:
    if get_class_name(obj) == target_type:
        return True
    else:
        return False


# Function to get the class name
# def get_class_name(cls: Type | object) -> str:
#     # If the input is an object, get its class first
#     if not isinstance(cls, type):
#         cls = cls.__class__

#     # Return the class name
#     return cls.__name__


# def validate_class_object(obj) -> bool:
#     if obj.__class__.__name__ in ['dict', 'list']:
#         return False

#     # Check if the object is an instance of a TypedDict
#     if isinstance(obj, dict) and hasattr(obj, "__annotations__"):
#         for base in obj.__class__.__bases__:
#             if isinstance(base.__dict__, MappingProxyType) and "__annotations__" in base.__dict__:
#                 return bool(obj.__class__.__name__)

#     return bool(obj.__class__.__name__)


def print_types_recursive(obj):
    if isinstance(obj, list):
        for item in obj:
            print_types_recursive(item)
    else:
        print(type(obj))


__all__ = [
    "check_object_type",
    "get_class_name",
    "print_types_recursive",
]

# Example usage
if __name__ == "__main__":
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
