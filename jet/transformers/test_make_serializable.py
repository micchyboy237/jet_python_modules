import json
import pytest
import abc
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar
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

# Custom class to simulate Protobuf Descriptor
class MockDescriptor:
    def __init__(self):
        self.__class__.__name__ = "Descriptor"
        self.__class__.__module__ = "google._upb._message"
    
    def __iter__(self):
        raise TypeError(f"'{self.__class__.__name__}' object is not iterable")

# Custom class to simulate RepeatedCompositeContainer
class MockRepeatedCompositeContainer:
    def __init__(self, items):
        self.__class__.__name__ = "RepeatedCompositeContainer"
        self.__class__.__module__ = "google._upb._message"
        self.items = items

    def __iter__(self):
        return iter(self.items)

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
    input_data = 42
    expected = 42
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_float():
    input_data = 3.14
    expected = 3.14
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_boolean():
    input_data = True
    expected = True
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_none():
    input_data = None
    expected = None
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_string():
    input_data = "hello world"
    expected = "hello world"
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_enum():
    input_data = SampleEnum.VALUE_ONE
    expected = "one"
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_np_integer():
    input_data = np.int64(100)
    expected = 100
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_np_float():
    input_data = np.float64(3.14)
    expected = 3.14
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_np_array():
    input_data = np.array([1, 2, 3, 4])
    expected = [1, 2, 3, 4]
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_bytes(sample_bytes):
    input_data = sample_bytes
    expected = {"key": "value"}
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_bytes_invalid(invalid_bytes):
    input_data = invalid_bytes
    expected = "gIGC"
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_class():
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
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_dict():
    input_data = {"key1": 1, "key2": 2}
    expected = {"key1": 1, "key2": 2}
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_dict_with_bytes(sample_bytes):
    input_data = {"key": sample_bytes}
    expected = {"key": {"key": "value"}}
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_dict_with_nested_bytes(sample_bytes):
    input_data = {"nested_key": sample_bytes}
    expected = {"nested_key": {"key": "value"}}
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_list():
    input_data = [1, 2, 3, 4]
    expected = [1, 2, 3, 4]
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_set():
    input_data = {1, 2, 3, 4}
    expected = [1, 2, 3, 4]
    result = make_serializable(input_data)
    assert sorted(result) == sorted(expected), f"Expected {expected}, but got {result}"

def test_serializable_tuple():
    input_data = (1, 2, 3, 4)
    expected = [1, 2, 3, 4]
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_model():
    input_data = SampleModel(name="Test", value=42)
    expected = {"name": "Test", "value": 42}
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_complex_structure(complex_structure):
    input_data = complex_structure["input"]
    expected = complex_structure["expected"]
    result = make_serializable(input_data)
    for key in result.keys():
        if key == "set":
            assert sorted(result[key]) == sorted(expected[key]), f"Error with {key}: Expected {expected[key]}, but got {result[key]}"
        else:
            assert result[key] == expected[key], f"Error with {key}: Expected {expected[key]}, but got {result[key]}"

def test_serializable_dict_with_string_keys():
    input_data = {
        0: ["Sample 1", "Sample 2"],
        1: ["Sample 3"]
    }
    expected = {
        "0": ["Sample 1", "Sample 2"],
        "1": ["Sample 3"]
    }
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_callable():
    def sample_function():
        return "I am a function"
    class SampleClass:
        def sample_method(self):
            return "I am a method"
    instance = SampleClass()
    input_data = {
        "function": sample_function,
        "builtin": len,
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
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_callable_class_instance():
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
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_circular_reference():
    class CircularClass:
        def __init__(self):
            self.name = "Test"
            self.self_ref = None
    obj = CircularClass()
    obj.self_ref = obj
    expected = {"name": "Test", "self_ref": "<CircularClass object>"}
    result = make_serializable(obj)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_protobuf_descriptor():
    input_data = MockDescriptor()
    expected = "Descriptor"
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_serializable_coref_chains():
    input_data = MockRepeatedCompositeContainer([
        {
            "DESCRIPTOR": MockDescriptor(),
            "chainID": 4,
            "mention": [
                {
                    "mentionID": 0,
                    "mentionType": "PROPER",
                    "number": "SINGULAR",
                    "gender": "MALE",
                    "animacy": "ANIMATE",
                    "beginIndex": 0,
                    "endIndex": 2,
                    "headIndex": 1,
                    "sentenceIndex": 0,
                    "position": 1
                },
                {
                    "mentionID": 2,
                    "mentionType": "PROPER",
                    "number": "SINGULAR",
                    "gender": "MALE",
                    "animacy": "ANIMATE",
                    "beginIndex": 0,
                    "endIndex": 1,
                    "headIndex": 0,
                    "sentenceIndex": 1,
                    "position": 1
                },
                {
                    "mentionID": 4,
                    "mentionType": "PRONOMINAL",
                    "number": "SINGULAR",
                    "gender": "MALE",
                    "animacy": "ANIMATE",
                    "beginIndex": 0,
                    "endIndex": 1,
                    "headIndex": 0,
                    "sentenceIndex": 2,
                    "position": 1
                }
            ],
            "representative": 0
        }
    ])
    expected = [
        {
            "descriptor": "Descriptor",
            "chain_id": 4,
            "mention": [
                {
                    "mention_id": 0,
                    "mention_type": "PROPER",
                    "number": "SINGULAR",
                    "gender": "MALE",
                    "animacy": "ANIMATE",
                    "begin_index": 0,
                    "end_index": 2,
                    "head_index": 1,
                    "sentence_index": 0,
                    "position": 1
                },
                {
                    "mention_id": 2,
                    "mention_type": "PROPER",
                    "number": "SINGULAR",
                    "gender": "MALE",
                    "animacy": "ANIMATE",
                    "begin_index": 0,
                    "end_index": 1,
                    "head_index": 0,
                    "sentence_index": 1,
                    "position": 1
                },
                {
                    "mention_id": 4,
                    "mention_type": "PRONOMINAL",
                    "number": "SINGULAR",
                    "gender": "MALE",
                    "animacy": "ANIMATE",
                    "begin_index": 0,
                    "end_index": 1,
                    "head_index": 0,
                    "sentence_index": 2,
                    "position": 1
                }
            ],
            "representative": 0
        }
    ]
    result = make_serializable(input_data)
    assert result == expected, f"Expected {expected}, but got {result}"
    try:
        json.dumps(result)
    except TypeError as e:
        pytest.fail(f"Result is not JSON-serializable: {str(e)}")

def test_serializable_infinity_and_nan():
    # Given various infinite and NaN values from Python and NumPy
    input_data = {
        "pos_inf": float("inf"),
        "neg_inf": float("-inf"),
        "nan": float("nan"),
        "np_pos_inf": np.inf,
        "np_neg_inf": -np.inf,
        "np_nan": np.nan,
        "np_float_inf": np.float64("inf"),
        "mixed": [1.0, float("inf"), float("-inf"), np.nan, -np.inf],
    }

    # When
    result = make_serializable(input_data)

    # Then
    expected = {
        "pos_inf": "Infinity",
        "neg_inf": "-Infinity",
        "nan": "NaN",
        "np_pos_inf": "Infinity",
        "np_neg_inf": "-Infinity",
        "np_nan": "NaN",
        "np_float_inf": "Infinity",
        "mixed": [1.0, "Infinity", "-Infinity", "NaN", "-Infinity"],
    }
    assert result == expected
    # Extra safety: must be JSON-serializable
    json.dumps(result)  # raises if not

def test_serializable_plain_class_instance_with_nested_objects():
    # Given: a plain class (no dataclass, no BaseModel, no custom __dict__) 
    #        containing mixed attributes and a nested instance of another plain class
    class Address:
        def __init__(self):
            self.street = "123 Main St"
            self.city = "Springfield"
            self._secret = "hidden"
            self.empty_list = []
            self.none_value = None

    class Person:
        def __init__(self):
            self.name = "Alice"
            self.age = 30
            self.is_active = True
            self.address = Address()
            self._password = "s3cr3t"
            self.hobbies = []           # empty → should be filtered
            self.notes = None           # None → should be filtered

    input_data = Person()

    # Expected: only non-empty, non-private attributes are kept
    #           nested objects are recursively serialized
    #           keys are converted to snake_case
    expected = {
        "name": "Alice",
        "age": 30,
        "is_active": True,
        "address": {
            "street": "123 Main St",
            "city": "Springfield"
        }
    }

    # When
    result = make_serializable(input_data)

    # Then
    assert result == expected, f"Expected {expected!r} but got {result!r}"

@pytest.fixture(autouse=True)
def cleanup():
    yield
