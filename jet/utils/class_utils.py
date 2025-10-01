from typing import Any, Dict
import inspect
from typing import Iterable, Type

from jet.logger import logger


# Function to check if an object is an instance of a user-defined class
def is_class_instance(obj):
    return isinstance(obj, object) and not isinstance(obj, dict)


# Function to check if an object is a dictionary
def is_dictionary(obj):
    # Ensures it's exactly a dictionary, not a subclass
    return type(obj) is dict


def class_to_string(cls: Type | object) -> str:
    # If the input is an object, get its class first
    if not isinstance(cls, type):
        cls = cls.__class__

    # Get the source code of the class
    class_source = inspect.getsource(cls)

    # Strip leading and trailing whitespace to match desired formatting
    return class_source.strip()


def validate_class(obj, expected_class):
    """
    Validates that an object is an instance of a specific class or subclass.

    :param obj: The object to validate.
    :param expected_class: The class or tuple of classes to check against.
    :raises TypeError: If the object is not an instance of the expected class.
    """
    if not isinstance(obj, expected_class):
        raise TypeError(f"Expected instance of {
                        expected_class}, but got {type(obj)}")


def get_class_name(obj, full_class_name=False, expected_class=None):
    """
    Returns the name of the class of an object, with optional validation.

    :param obj: The object whose class name is to be retrieved.
    :param full_class_name: If True, return the full class name including module. Default is False.
    :param expected_class: The class or tuple of classes to validate the object against. Default is None (no validation).
    :return: Class name (string) or full class name (string) depending on the `full_class_name` flag.
    :raises TypeError: If the object is not an instance of the expected class.
    """
    if expected_class:
        validate_class(obj, expected_class)

    if not isinstance(obj, object):
        raise TypeError(f"Expected an object, got {type(obj)}")

    # Get the class of the object
    class_name = obj.__class__.__name__

    if full_class_name:
        # Get full class name with module
        class_name = f"{obj.__class__.__module__}.{class_name}"

    return class_name


def validate_iterable_class(obj: Any, expected_class: Type) -> None:
    """
    Validates that the object is both iterable and an instance of the expected class.

    :param obj: The object to validate.
    :param expected_class: The class to check the object against.
    :raises TypeError: If the object is not iterable or not of the expected class.
    """
    # Check if the object is iterable
    if not isinstance(obj, Iterable):
        raise TypeError(f"Expected an iterable object, but got {type(obj)}")

    # Check if the object is an instance of the expected class
    if not isinstance(obj, expected_class):
        raise TypeError(f"Expected instance of {
                        expected_class}, but got {type(obj)}")


def get_iterable_class_name(obj: Any, full_class_name=False) -> str:
    """
    Returns the class name of an iterable object, with optional full class name including the module.

    :param obj: The iterable object to retrieve the class name from.
    :param full_class_name: If True, return the full class name including the module. Default is False.
    :return: Class name (string) or full class name (string) depending on `full_class_name` flag.
    :raises TypeError: If the object is not iterable.
    """
    # Check if the object is iterable
    if not isinstance(obj, Iterable):
        raise TypeError(f"Expected an iterable object, but got {type(obj)}")

    # Get the class of the object
    class_name = obj.__class__.__name__

    if full_class_name:
        # Get full class name with module
        class_name = f"{obj.__class__.__module__}.{class_name}"

    return class_name


def get_builtin_attributes(obj: Any) -> list[str]:
    """Returns built-in attributes."""
    built_in_attrs = set(dir(type(obj)))
    return list(built_in_attrs)


def get_non_empty_attributes(obj: Any) -> Dict[str, Any]:
    """Returns non-callable attributes that are not empty or private (_attr)."""
    attributes = {}
    for attr in dir(obj):
        if not attr.startswith('_'):  # Exclude private and dunder attributes
            try:
                value = getattr(obj, attr, None)  # Safely get attribute
                if value is not None and value not in ['', [], {}, ()] and not callable(value):
                    attributes[attr] = value
            except AttributeError as e:
                print(f"Skipping attribute {attr} due to error: {e}")
            except Exception as e:
                print(f"Unexpected error for attribute {attr}: {e}")
    return attributes


def get_non_empty_primitive_attributes(obj: Any) -> Dict[str, Any]:
    """
    Returns non-callable, non-empty primitive attributes (int, float, str, bool, etc.) that are not private (_attr).

    :param obj: The object to inspect.
    :return: Dictionary of non-empty primitive attributes.
    """
    attributes = {}
    for attr in dir(obj):
        if not attr.startswith('_'):  # Exclude private and dunder attributes
            try:
                value = getattr(obj, attr, None)
                if (
                    value is not None
                    and value not in ['', [], {}, ()]
                    and not callable(value)
                    and isinstance(value, (int, float, str, bool))
                ):
                    attributes[attr] = value
            except AttributeError as e:
                logger.debug(f"Skipping attribute {attr} due to error: {e}")
            except Exception as e:
                logger.debug(f"Unexpected error for attribute {attr}: {e}")
    return attributes


def get_non_empty_object_attributes(obj: Any) -> Dict[str, Any]:
    """
    Returns non-callable, non-empty object attributes (instances of user-defined classes, lists, dicts, etc.)
    that are not private (_attr).

    :param obj: The object to inspect.
    :return: Dictionary of non-empty object attributes.
    """
    attributes = {}
    for attr in dir(obj):
        if not attr.startswith('_'):  # Exclude private and dunder attributes
            try:
                value = getattr(obj, attr, None)
                if (
                    value is not None
                    and value not in ['', [], {}, ()]
                    and not callable(value)
                    and not isinstance(value, (int, float, str, bool))
                ):
                    attributes[attr] = value
            except AttributeError as e:
                logger.debug(f"Skipping attribute {attr} due to error: {e}")
            except Exception as e:
                logger.debug(f"Unexpected error for attribute {attr}: {e}")
    return attributes


def get_internal_attributes(obj: Any) -> Dict[str, Any]:
    """Returns private/internal attributes (_attr) but excludes built-in dunder attributes (__attr__)."""
    attributes = {}
    built_in_attrs = set(dir(type(obj)))  # Built-in attributes of the class

    for attr in dir(obj):
        # Exclude built-in dunder attributes
        if attr.startswith('_') and attr not in built_in_attrs:
            try:
                attributes[attr] = getattr(obj, attr, None)
            except Exception as e:
                print(f"Skipping attribute {attr} due to error: {e}")

    return attributes


def get_callable_attributes(obj: Any) -> Dict[str, Any]:
    """Returns callable attributes (methods) that are not private (_attr)."""
    callables = {}
    for attr in dir(obj):
        if not attr.startswith('_'):  # Exclude private methods
            try:
                value = getattr(obj, attr, None)
                if callable(value):
                    callables[attr] = value
            except Exception as e:
                print(f"Skipping attribute {attr} due to error: {e}")
    return callables


def get_non_callable_attributes(obj: Any) -> Dict[str, Any]:
    attributes = {}
    for attr in dir(obj):
        value = getattr(obj, attr)
        if not callable(value):  # Exclude methods
            attributes[attr] = value
    return attributes


# Real-world usage examples
# Example usage 1:
if __name__ == "__main__":
    logger.info("Example usage 1:")
    # Example Classes

    class Animal:
        def __init__(self, name: str):
            self.name = name

    class Dog(Animal):
        def __init__(self, name: str, breed: str):
            super().__init__(name)
            self.breed = breed

    class Cat(Animal):
        def __init__(self, name: str, color: str):
            super().__init__(name)
            self.color = color

    # Instantiate objects
    my_dog = Dog(name="Rex", breed="German Shepherd")
    my_cat = Cat(name="Whiskers", color="Tabby")

    # Real-world assertion examples:

    # Test validate_class
    try:
        # Assert valid instance of Dog
        validate_class(my_dog, Dog)  # Should pass
        print("Assertion passed: my_dog is an instance of Dog.")

        # Assert valid instance of Animal (Dog is subclass of Animal)
        validate_class(my_dog, Animal)  # Should pass
        print("Assertion passed: my_dog is an instance of Animal.")

        # Assert invalid instance type (my_dog is not a Cat)
        validate_class(my_dog, Cat)  # Should raise TypeError
    except TypeError as e:
        assert get_class_name(e) == "TypeError"

    # Test get_class_name with validation
    try:
        # Assert valid class name for Dog
        assert get_class_name(my_dog) == "Dog"
        print("Assertion passed: Class name of my_dog is 'Dog'.")

        # Assert valid class name for Animal (Dog is a subclass of Animal)
        assert get_class_name(my_dog, expected_class=Animal) == "Dog"
        print("Assertion passed: Class name of my_dog with validation is 'Dog'.")

        # Assert full class name (module + class)
        assert get_class_name(my_dog, full_class_name=True) == "__main__.Dog"
        print("Assertion passed: Full class name of my_dog is '__main__.Dog'.")

        # Assert class name for Cat
        assert get_class_name(my_cat) == "Cat"
        print("Assertion passed: Class name of my_cat is 'Cat'.")

        # Test with incorrect expected_class
        try:
            # Should raise TypeError
            assert get_class_name(my_dog, expected_class=str)
        except TypeError as e:
            assert get_class_name(e) == "TypeError"

    except AssertionError as e:
        assert get_class_name(e) == "AssertionError"

# Example usage 2:
if __name__ == "__main__":
    logger.info("Example usage 2:")
    from pydantic import BaseModel
    from typing import Optional

    class CodeSummary(BaseModel):
        features: list[str]
        use_cases: list[str]
        additional_info: Optional[str] = None

    # Stringify the class itself
    class_stringified_version = class_to_string(CodeSummary)
    print(class_stringified_version)

    # Create an object of the class
    code_summary_obj = CodeSummary(
        features=["Sample feature"],
        use_cases=["Sample use case"],
    )

    # Stringify the class of the object
    obj_stringified_version = class_to_string(code_summary_obj)
    print(obj_stringified_version)

    assert class_stringified_version == obj_stringified_version


# Example usage 3:
if __name__ == "__main__":
    logger.info("Example usage 3:")
    # Example Classes

    class Animal:
        def __init__(self, name: str):
            self.name = name

    class Dog(Animal):
        def __init__(self, name: str, breed: str):
            super().__init__(name)
            self.breed = breed

    # Create an iterable object
    dog_list = [Dog(name="Rex", breed="German Shepherd"),
                Dog(name="Max", breed="Bulldog")]

    # Validating the iterable and its class
    try:
        # Should pass (dog_list is a list)
        validate_iterable_class(dog_list, list)
        print("Assertion passed: dog_list is an iterable and an instance of list.")

        # Validate if the objects inside the iterable are instances of Dog
        for dog in dog_list:
            # Should pass (dog is an instance of Dog)
            validate_iterable_class(dog, Dog)
            print("Assertion passed: dog is an instance of Dog.")
    except TypeError as e:
        assert get_class_name(e) == "TypeError"

    # Test get_iterable_class_name with validation
    try:
        # Assert valid class name for the iterable (list)
        assert get_iterable_class_name(dog_list) == "list"
        print("Assertion passed: Class name of dog_list is 'list'.")

        # Assert full class name for the iterable (list)
        assert ".list" in get_iterable_class_name(
            dog_list, full_class_name=True)
        print("Assertion passed: Full class name of dog_list is '*.list'.")

        # Assert class name of individual objects inside the iterable (Dog)
        for dog in dog_list:
            assert get_iterable_class_name(dog) == "Dog"
            print("Assertion passed: Class name of dog is 'Dog'.")
    except TypeError as e:
        assert get_class_name(e) == "TypeError"


__all__ = [
    "is_class_instance",
    "is_dictionary",
    "class_to_string",
    "validate_class",
    "get_class_name",
    "validate_iterable_class",
    "get_iterable_class_name",
    "get_builtin_attributes",
    "get_non_empty_attributes",
    "get_non_empty_primitive_attributes",
    "get_non_empty_object_attributes",
    "get_internal_attributes",
    "get_callable_attributes",
    "get_non_callable_attributes",
]
