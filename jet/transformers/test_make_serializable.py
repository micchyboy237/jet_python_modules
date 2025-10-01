import pytest
import abc
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar
import numpy as np
from pydantic import BaseModel
from jet.transformers.object import make_serializable

# Enum example for testing
class SampleEnum(Enum):
    VALUE_ONE = "one"
    VALUE_TWO = "two"

# Example Pydantic BaseModel
class SampleModel(BaseModel):
    name: str
    value: int

@pytest.fixture
def sample_bytes():
    return b'{"key": "value"}'

@pytest.fixture
def invalid_bytes():
    return b'\x80\x81\x82'

@pytest.fixture
def complex_structure():
    byte_val_dict = {"model": "llama3.2:latest"}
    byte_val = b'{"model": "llama3.2:latest"}'
    nested_bytes_val = {
        "key": byte_val,
        "nested_bytes": {
            "nested_key": byte_val
        }
    }
    nested_bytes_val_dict = {
        "key": byte_val_dict,
        "nested_bytes": {
            "nested_key": byte_val_dict
        }
    }
    return {
        "input": {
            "list": [4, 2, 3, 2, 5],
            "list_bytes": [byte_val, nested_bytes_val],
            "tuple": (4, 2, 3, 2, 5),
            "tuple_bytes": (byte_val, nested_bytes_val),
            "set": {4, 2, 3, 2, 5},
            "set_bytes": {byte_val, byte_val},
            "dict_bytes": nested_bytes_val
        },
        "expected": {
            "list": [4, 2, 3, 2, 5],
            "list_bytes": [byte_val_dict, nested_bytes_val_dict],
            "tuple": [4, 2, 3, 2, 5],
            "tuple_bytes": [byte_val_dict, nested_bytes_val_dict],
            "set": [2, 3, 4, 5],
            "set_bytes": [byte_val_dict],
            "dict_bytes": nested_bytes_val_dict
        }
    }

def test_serializable_int():
    # Given: An integer value
    input_data = 42
    expected = 42
    
    # When: We call make_serializable on the integer
    result = make_serializable(input_data)
    
    # Then: The result should match the expected integer
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_float():
    # Given: A float value
    input_data = 3.14
    expected = 3.14
    
    # When: We call make_serializable on the float
    result = make_serializable(input_data)
    
    # Then: The result should match the expected float
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_boolean():
    # Given: A boolean value
    input_data = True
    expected = True
    
    # When: We call make_serializable on the boolean
    result = make_serializable(input_data)
    
    # Then: The result should match the expected boolean
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_none():
    # Given: A None value
    input_data = None
    expected = None
    
    # When: We call make_serializable on None
    result = make_serializable(input_data)
    
    # Then: The result should match the expected None
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_string():
    # Given: A string value
    input_data = "hello world"
    expected = "hello world"
    
    # When: We call make_serializable on the string
    result = make_serializable(input_data)
    
    # Then: The result should match the expected string
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_enum():
    # Given: An Enum value
    input_data = SampleEnum.VALUE_ONE
    expected = "one"
    
    # When: We call make_serializable on the Enum
    result = make_serializable(input_data)
    
    # Then: The result should match the Enum's value
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_np_integer():
    # Given: A NumPy integer
    input_data = np.int64(100)
    expected = 100
    
    # When: We call make_serializable on the NumPy integer
    result = make_serializable(input_data)
    
    # Then: The result should match the expected Python integer
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_np_float():
    # Given: A NumPy float
    input_data = np.float64(3.14)
    expected = 3.14
    
    # When: We call make_serializable on the NumPy float
    result = make_serializable(input_data)
    
    # Then: The result should match the expected Python float
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_np_array():
    # Given: A NumPy array
    input_data = np.array([1, 2, 3, 4])
    expected = [1, 2, 3, 4]
    
    # When: We call make_serializable on the NumPy array
    result = make_serializable(input_data)
    
    # Then: The result should match the expected list
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_bytes(sample_bytes):
    # Given: A JSON-encoded bytes object
    input_data = sample_bytes
    expected = {"key": "value"}
    
    # When: We call make_serializable on the bytes
    result = make_serializable(input_data)
    
    # Then: The result should match the expected JSON-decoded dictionary
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_bytes_invalid(invalid_bytes):
    # Given: An invalid bytes object that cannot be decoded to UTF-8
    input_data = invalid_bytes
    expected = "gIGC"  # Base64 encoded form
    
    # When: We call make_serializable on the invalid bytes
    result = make_serializable(input_data)
    
    # Then: The result should match the expected Base64 string
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_class():
    # Given: A custom class instance with attributes
    T = TypeVar("T", bound="Serializable")

    class Serializable(metaclass=abc.ABCMeta):
        @classmethod
        @abc.abstractmethod
        def from_state(cls: type[T], state) -> T:
            raise NotImplementedError()

    @dataclass
    class MessageData(Serializable):
        http_version: str

        def from_state(self, state):
            pass

    @dataclass
    class RequestData(MessageData):
        host: str
        port: int

    input_data = RequestData(
        host='localhost',
        port=8080,
        http_version="1.0"
    )
    expected = {
        "host": 'localhost',
        "http_version": "1.0",
        "port": 8080,
    }
    
    # When: We call make_serializable on the class instance
    result = make_serializable(input_data)
    
    # Then: The result should match the expected dictionary
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_dict():
    # Given: A dictionary with simple key-value pairs
    input_data = {"key1": 1, "key2": 2}
    expected = {"key1": 1, "key2": 2}
    
    # When: We call make_serializable on the dictionary
    result = make_serializable(input_data)
    
    # Then: The result should match the expected dictionary
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_dict_with_bytes(sample_bytes):
    # Given: A dictionary with a JSON-encoded bytes value
    input_data = {"key": sample_bytes}
    expected = {"key": {"key": "value"}}
    
    # When: We call make_serializable on the dictionary
    result = make_serializable(input_data)
    
    # Then: The result should match the expected dictionary
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_dict_with_nested_bytes(sample_bytes):
    # Given: A dictionary with a nested JSON-encoded bytes value
    input_data = {"nested_key": sample_bytes}
    expected = {"nested_key": {"key": "value"}}
    
    # When: We call make_serializable on the dictionary
    result = make_serializable(input_data)
    
    # Then: The result should match the expected dictionary
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_list():
    # Given: A list of integers
    input_data = [1, 2, 3, 4]
    expected = [1, 2, 3, 4]
    
    # When: We call make_serializable on the list
    result = make_serializable(input_data)
    
    # Then: The result should match the expected list
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_set():
    # Given: A set of integers
    input_data = {1, 2, 3, 4}
    expected = [1, 2, 3, 4]  # Sets are converted to lists
    
    # When: We call make_serializable on the set
    result = make_serializable(input_data)
    
    # Then: The result should match the expected list
    assert sorted(result) == sorted(expected), f"Expected {expected}, but got {result}"

def test_serializable_tuple():
    # Given: A tuple of integers
    input_data = (1, 2, 3, 4)
    expected = [1, 2, 3, 4]  # Tuples are converted to lists
    
    # When: We call make_serializable on the tuple
    result = make_serializable(input_data)
    
    # Then: The result should match the expected list
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_model():
    # Given: A Pydantic model instance
    input_data = SampleModel(name="Test", value=42)
    expected = {"name": "Test", "value": 42}
    
    # When: We call make_serializable on the Pydantic model
    result = make_serializable(input_data)
    
    # Then: The result should match the expected dictionary
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_complex_structure(complex_structure):
    # Given: A complex nested structure with lists, tuples, sets, and bytes
    input_data = complex_structure["input"]
    expected = complex_structure["expected"]
    
    # When: We call make_serializable on the complex structure
    result = make_serializable(input_data)
    
    # Then: The result should match the expected structure
    for key in result.keys():
        if key == "set":  # Handle set ordering
            assert sorted(result[key]) == sorted(expected[key]), f"Error with {key}: Expected {expected[key]}, but got {result[key]}"
        else:
            assert result[key] == expected[key], f"Error with {key}: Expected {expected[key]}, but got {result[key]}"

def test_serializable_dict_with_string_keys():
    # Given: A dictionary with non-string keys
    input_data = {
        0: ["Sample 1", "Sample 2"],
        1: ["Sample 3"]
    }
    expected = {
        "0": ["Sample 1", "Sample 2"],
        "1": ["Sample 3"]
    }
    
    # When: We call make_serializable on the dictionary
    result = make_serializable(input_data)
    
    # Then: The result should match the expected dictionary with string keys
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_callable():
    # Given: A dictionary containing function objects (function, built-in function, and method)
    def sample_function():
        return "I am a function"
    
    class SampleClass:
        def sample_method(self):
            return "I am a method"
    
    instance = SampleClass()
    
    input_data = {
        "function": sample_function,
        "builtin": len,  # Built-in function
        "method": instance.sample_method,
        "nested": {
            "function": sample_function
        }
    }
    expected = {
        "function": "<class 'function'>",
        "builtin": "<class 'builtin_function_or_method'>",
        "method": "<class 'method'>",
        "nested": {
            "function": "<class 'function'>"
        }
    }
    
    # When: We call make_serializable on the dictionary with functions
    result = make_serializable(input_data)
    
    # Then: The result should match the expected dictionary with function type strings
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_callable_class_instance():
    # Given: A class instance that implements __call__ to simulate objects like swarms.structs.agent.Agent
    class CallableClass:
        def __init__(self):
            self.name = "TestAgent"
            self.value = 42
        
        def __call__(self, *args, **kwargs):
            return "Called"
        
        def __dict__(self):
            return {"name": self.name, "value": self.value}
    
    input_data = {
        "self": CallableClass(),
        "nested": {
            "self": CallableClass()
        }
    }
    expected = {
        "self": {"name": "TestAgent", "value": 42},
        "nested": {
            "self": {"name": "TestAgent", "value": 42}
        }
    }
    
    # When: We call make_serializable on the dictionary with callable class instances
    result = make_serializable(input_data)
    
    # Then: The result should match the expected dictionary with attributes, not callable types
    assert result == expected, f"Expected {expected}, but got {result}"

@pytest.fixture(autouse=True)
def cleanup():
    yield
    # No specific cleanup required for these tests