from typing import Any, Dict
import inspect
from typing import Any, Iterable, Type

from jet.logger import logger


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


def get_non_empty_attributes(obj: Any) -> Dict[str, Any]:
    """
    Extracts the non-empty attributes of an object (excluding those starting with "_"
    and methods) and returns them in a dictionary, filtering out attributes with values
    that are considered empty or falsy.

    Args:
        obj: The object from which to extract attributes.

    Returns:
        A dictionary with attribute names as keys and their corresponding
        non-falsy values as values.
    """
    return {
        attr: value
        for attr in dir(obj)
        if not attr.startswith('_')
        and (value := getattr(obj, attr))
        and not callable(value)  # Exclude methods
    }


def get_internal_attributes(obj: Any) -> Dict[str, Any]:
    """
    Extracts the attributes of an object that start with "_" and returns them in a dictionary.

    Args:
        obj: The object from which to extract attributes.

    Returns:
        A dictionary with attribute names starting with "_" as keys and their corresponding values.
    """
    return {
        attr: getattr(obj, attr)
        for attr in dir(obj)
        if attr.startswith('_')
    }


def get_callable_attributes(obj: Any) -> Dict[str, Any]:
    """
    Extracts the callable attributes of an object and returns them in a dictionary.

    Args:
        obj: The object from which to extract callable attributes.

    Returns:
        A dictionary with attribute names as keys and the callable objects as values.
    """
    return {
        attr: getattr(obj, attr)
        for attr in dir(obj)
        if callable(getattr(obj, attr))  # Filter for callable attributes
        and not attr.startswith('_')
    }


__all__ = [
    "class_to_string",
    "validate_class",
    "get_class_name",
    "validate_iterable_class",
    "get_iterable_class_name",
    "get_non_empty_attributes",
    "get_internal_attributes",
    "get_callable_attributes",
]

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
            print(f"Assertion passed: dog is an instance of Dog.")
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
            print(f"Assertion passed: Class name of dog is 'Dog'.")
    except TypeError as e:
        assert get_class_name(e) == "TypeError"
