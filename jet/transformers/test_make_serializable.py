import abc
from dataclasses import dataclass
import json
from typing import TypeVar
import unittest
from enum import Enum
from jet.transformers.object import make_serializable
import numpy as np
from pydantic import BaseModel

# Enum example for testing


class SampleEnum(Enum):
    VALUE_ONE = "one"
    VALUE_TWO = "two"

# Example Pydantic BaseModel


class SampleModel(BaseModel):
    name: str
    value: int


class TestMakeSerializable(unittest.TestCase):

    def test_serializable_int(self):
        obj = 42
        result = make_serializable(obj)
        self.assertEqual(result, 42)

    def test_serializable_float(self):
        obj = 3.14
        result = make_serializable(obj)
        self.assertEqual(result, 3.14)

    def test_serializable_boolean(self):
        obj = True
        result = make_serializable(obj)
        self.assertEqual(result, True)

    def test_serializable_none(self):
        obj = None
        result = make_serializable(obj)
        self.assertEqual(result, None)

    def test_serializable_string(self):
        obj = "hello world"
        result = make_serializable(obj)
        self.assertEqual(result, "hello world")

    def test_serializable_enum(self):
        obj = SampleEnum.VALUE_ONE
        result = make_serializable(obj)
        self.assertEqual(result, "one")

    def test_serializable_np_integer(self):
        obj = np.int64(100)
        result = make_serializable(obj)
        self.assertEqual(result, 100)

    def test_serializable_np_float(self):
        obj = np.float64(3.14)
        result = make_serializable(obj)
        self.assertEqual(result, 3.14)

    def test_serializable_np_array(self):
        obj = np.array([1, 2, 3, 4])
        result = make_serializable(obj)
        self.assertEqual(result, [1, 2, 3, 4])

    def test_serializable_bytes(self):
        obj = b'{"key": "value"}'
        result = make_serializable(obj)
        self.assertEqual(json.dumps(result), json.dumps({"key": "value"}))

    def test_serializable_bytes_invalid(self):
        obj = b'\x80\x81\x82'
        result = make_serializable(obj)
        self.assertEqual(result, "gIGC")  # Base64 encoded form

    def test_serializable_class(self):
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
        obj = RequestData(
            host='localhost',
            port=8080,
            http_version="1.0"
        )
        result = make_serializable(obj)
        self.assertEqual(
            json.dumps(result),
            json.dumps({
                "host": 'localhost',
                "http_version": "1.0",
                "port": 8080,
            }))

    def test_serializable_dict(self):
        obj = {"key1": 1, "key2": 2}
        result = make_serializable(obj)
        self.assertEqual(json.dumps(result),
                         json.dumps({"key1": 1, "key2": 2}))

    def test_serializable_dict_with_bytes(self):
        obj = {"key": b'{"model": "llama3.2:latest"}'}
        result = make_serializable(obj)
        self.assertEqual(json.dumps(result), json.dumps(
            {"key": {"model": "llama3.2:latest"}}))

    def test_serializable_dict_with_nested_bytes(self):
        obj = {"nested_key": b'{"nested": "data"}'}
        result = make_serializable(obj)
        self.assertEqual(json.dumps(result), json.dumps(
            {"nested_key": {"nested": "data"}}))

    def test_serializable_list(self):
        obj = [1, 2, 3, 4]
        result = make_serializable(obj)
        self.assertEqual(result, [1, 2, 3, 4])

    def test_serializable_set(self):
        obj = {1, 2, 3, 4}
        result = make_serializable(obj)
        self.assertEqual(result, [1, 2, 3, 4])  # Sets are converted to lists

    def test_serializable_tuple(self):
        obj = (1, 2, 3, 4)
        result = make_serializable(obj)
        self.assertEqual(result, [1, 2, 3, 4])  # Tuples are converted to lists

    def test_serializable_model(self):
        obj = SampleModel(name="Test", value=42)
        result = make_serializable(obj)
        self.assertEqual(result, {"name": "Test", "value": 42})

    def test_serializable_complex_structure(self):
        byte_val_dict = {"model": "llama3.2:latest"}
        byte_val = b'{"model": "llama3.2:latest"}'
        nested_bytes_val = {
            "key":  byte_val,
            "nested_bytes": {
                "nested_key":  byte_val
            }
        }
        nested_bytes_val_dict = {
            "key":  byte_val_dict,
            "nested_bytes": {
                "nested_key":  byte_val_dict
            }
        }

        obj = {
            "list": [4, 2, 3, 2, 5],
            "list_bytes": [byte_val, nested_bytes_val],
            "tuple": (4, 2, 3, 2, 5),
            "tuple_bytes": (byte_val, nested_bytes_val),
            "set": {4, 2, 3, 2, 5},
            "set_bytes": {byte_val, byte_val},
            "dict_bytes": nested_bytes_val
        }
        result = make_serializable(obj)
        expected = {
            "list": [4, 2, 3, 2, 5],
            "list_bytes": [byte_val_dict, nested_bytes_val_dict],
            "tuple": [4, 2, 3, 2, 5],
            "tuple_bytes": [byte_val_dict, nested_bytes_val_dict],
            "set": [2, 3, 4, 5],
            "set_bytes": [byte_val_dict],
            "dict_bytes": nested_bytes_val_dict
        }

        for key in result.keys():
            self.assertEqual(result[key], expected[key], f"Error with {key}")

    def test_serializable_dict_with_string_keys(self):
        obj = {
            0: ["Sample 1", "Sample 2"],
            1: ["Sample 3"]
        }
        result = make_serializable(obj)
        expected = {
            "0": ["Sample 1", "Sample 2"],
            "1": ["Sample 3"]
        }
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
