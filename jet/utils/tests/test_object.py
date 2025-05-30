import json
import unittest
from typing import TypedDict
from dataclasses import dataclass
from jet.utils.object import check_object_type, get_class_name, extract_values_by_paths


class TestCheckObjectType(unittest.TestCase):
    def test_primitive_types(self):
        self.assertTrue(check_object_type(42, 'int'))
        self.assertTrue(check_object_type("Hello, World!", 'str'))
        self.assertTrue(check_object_type([1, 2, 3], 'list'))
        self.assertTrue(check_object_type(
            {"name": "Alice", "age": 25}, 'dict'))

    def test_typed_dict(self):
        class PersonTypedDict(dict):
            name: str
            age: int

        person_typed_dict = PersonTypedDict(name="Alice", age=25)
        self.assertTrue(check_object_type(
            person_typed_dict, 'PersonTypedDict'))

    def test_dataclass(self):
        @dataclass
        class PersonDataClass:
            name: str
            age: int

        person_dataclass = PersonDataClass(name="John", age=30)
        self.assertTrue(check_object_type(person_dataclass, 'PersonDataClass'))

    def test_regular_class_with_init(self):
        class PersonWithInit:
            def __init__(self, name: str, age: int):
                self.name = name
                self.age = age

        person_with_init = PersonWithInit(name="Bob", age=40)
        self.assertTrue(check_object_type(person_with_init, 'PersonWithInit'))

    def test_regular_class_without_init(self):
        class PersonBlueprint:
            name: str
            age: int

        person_blueprint = PersonBlueprint()
        self.assertTrue(check_object_type(person_blueprint, 'PersonBlueprint'))


class TestGetClassName(unittest.TestCase):
    def test_normal_object(self):
        class CustomClass:
            pass

        obj = CustomClass()
        self.assertEqual(get_class_name(obj), "CustomClass")

    def test_builtin_object(self):
        obj = [1, 2, 3]
        self.assertEqual(get_class_name(obj), "list")

        obj = {"a": 1, "b": 2}
        self.assertEqual(get_class_name(obj), "dict")

    def test_typed_dict(self):
        class ExampleTypedDict(dict):
            key1: str
            key2: int

        obj = ExampleTypedDict(key1="value1", key2=123)
        self.assertEqual(get_class_name(obj), "ExampleTypedDict")

    def test_builtin_exception(self):
        try:
            raise ValueError("Example error")
        except ValueError as e:
            self.assertEqual(get_class_name(e), "ValueError")

    def test_invalid_dict_with_annotations(self):
        class FakeTypedDict(dict):
            __annotations__ = {"key1": str, "key2": int}

        obj = FakeTypedDict()
        self.assertEqual(get_class_name(obj), "FakeTypedDict")

    def test_non_typed_dict(self):
        obj = {"key1": "value1", "key2": 123}
        self.assertEqual(get_class_name(obj), "dict")


class TestExtractValuesByPaths(unittest.TestCase):
    def test_basic_user_info(self):
        data = {"user": {"name": "Alice", "age": 30, "email": "alice@example.com"}}
        paths = ["user.name", "user.age"]
        expected = {"user": {"name": "Alice", "age": 30}}
        expected_flattened = {"name": "Alice", "age": 30}
        self.assertEqual(extract_values_by_paths(data, paths), expected)
        self.assertEqual(extract_values_by_paths(
            data, paths, True), expected_flattened)

    def test_nested_order_details(self):
        data = {"order": {"id": 123, "items": {"item1": {"name": "Laptop",
                                                         "price": 999}, "item2": {"name": "Mouse", "price": 49}}, "total": 1048}}
        paths = ["order.items.item1.name", "order.total"]
        expected = {
            "order": {"items": {"item1": {"name": "Laptop"}}, "total": 1048}}
        expected_flattened = {"name": "Laptop", "total": 1048}
        self.assertEqual(extract_values_by_paths(data, paths), expected)
        self.assertEqual(extract_values_by_paths(
            data, paths, True), expected_flattened)

    def test_missing_keys(self):
        data = {"profile": {"username": "john_doe", "location": "USA"}}
        paths = ["profile.username", "profile.email", "settings.theme"]
        expected = {"profile": {"username": "john_doe"}}
        expected_flattened = {"username": "john_doe"}
        self.assertEqual(extract_values_by_paths(data, paths), expected)
        self.assertEqual(extract_values_by_paths(
            data, paths, True), expected_flattened)

    def test_partial_user_address(self):
        data = {"user": {"address": {"city": "New York",
                                     "zipcode": "10001"}, "phone": "123-456-7890"}}
        paths = ["user.address.city", "user.address.zipcode", "user.phone"]
        expected = {"user": {"address": {"city": "New York",
                                         "zipcode": "10001"}, "phone": "123-456-7890"}}
        expected_flattened = {"city": "New York",
                              "zipcode": "10001", "phone": "123-456-7890"}
        self.assertEqual(extract_values_by_paths(data, paths), expected)
        self.assertEqual(extract_values_by_paths(
            data, paths, True), expected_flattened)

    def test_root_level_keys(self):
        data = {"id": 1, "name": "Sample", "details": {"category": "Tech"}}
        paths = ["id", "details.category"]
        expected = {"id": 1, "details": {"category": "Tech"}}
        expected_flattened = {"id": 1, "category": "Tech"}
        self.assertEqual(extract_values_by_paths(data, paths), expected)
        self.assertEqual(extract_values_by_paths(
            data, paths, True), expected_flattened)

    def test_duplicate_keys_error(self):
        data = {"order": {"item": {"name": "Laptop", "total": 100}, "total": 200}}
        paths = ["order.item.name", "order.item.total", "order.total"]
        with self.assertRaises(ValueError):
            extract_values_by_paths(data, paths, True)


if __name__ == "__main__":
    unittest.main()
