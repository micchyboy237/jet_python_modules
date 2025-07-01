import pytest
from jet.utils.print_utils import print_dict_types


class TestPrintDictTypes:
    def test_simple_dictionary(self):
        """Test with a simple dictionary containing basic types"""
        # Given
        input_data = {
            "name": "John",
            "age": 30,
            "active": True,
            "score": 95.5
        }
        expected = [
            "name: str",
            "age: int",
            "active: bool",
            "score: float"
        ]

        # When
        result = print_dict_types(input_data)

        # Then
        assert result == expected

    def test_nested_dictionary(self):
        """Test with nested dictionaries"""
        # Given
        input_data = {
            "user": {
                "name": "Alice",
                "details": {
                    "age": 25,
                    "city": "New York"
                }
            }
        }
        expected = [
            "user: dict",
            "  user.name: str",
            "  user.details: dict",
            "    user.details.age: int",
            "    user.details.city: str"
        ]

        # When
        result = print_dict_types(input_data)

        # Then
        assert result == expected

    def test_dictionary_with_list(self):
        """Test with dictionary containing a list of mixed types"""
        # Given
        input_data = {
            "items": [1, "text", True, {"key": 42}],
            "name": "test"
        }
        expected = [
            "items: list",
            "  items: int",
            "  items: str",
            "  items: bool",
            "  items: dict",
            "    items.key: int",
            "name: str"
        ]

        # When
        result = print_dict_types(input_data)

        # Then
        assert result == expected

    def test_nested_list_and_dict(self):
        """Test with deeply nested lists and dictionaries"""
        # Given
        input_data = {
            "data": [
                {"id": 1, "values": [10, 20]},
                {"id": 2, "values": ["a", {"x": False}]}
            ]
        }
        expected = [
            "data: list",
            "  data: dict",
            "    data.id: int",
            "    data.values: list",
            "      data.values: int",
            "      data.values: int",
            "  data: dict",
            "    data.id: int",
            "    data.values: list",
            "      data.values: str",
            "      data.values: dict",
            "        data.values.x: bool"
        ]

        # When
        result = print_dict_types(input_data)

        # Then
        assert result == expected

    def test_empty_structures(self):
        """Test with empty dictionary and list"""
        # Given
        input_data = {
            "empty_dict": {},
            "empty_list": [],
            "normal": 42
        }
        expected = [
            "empty_dict: dict",
            "empty_list: list",
            "normal: int"
        ]

        # When
        result = print_dict_types(input_data)

        # Then
        assert result == expected

    def test_non_dict_input(self):
        """Test with non-dictionary input (single list)"""
        # Given
        input_data = [1, "two", [3, 4]]
        expected = [
            ": list",
            "  : int",
            "  : str",
            "  : list",
            "    : int",
            "    : int"
        ]

        # When
        result = print_dict_types(input_data)

        # Then
        assert result == expected

    def test_list_with_uniform_dicts(self):
        """Test with a list of dictionaries with identical structure"""
        # Given
        input_data = {
            "raw_tokens_sequential": [
                {"id": 1, "type": "string", "content": "hello", "checked": None},
                {"id": 2, "type": "number", "content": "123", "checked": None}
            ]
        }
        expected = [
            "raw_tokens_sequential[]: list[dict]",
            "  raw_tokens_sequential[].id: int",
            "  raw_tokens_sequential[].type: str",
            "  raw_tokens_sequential[].content: str",
            "  raw_tokens_sequential[].checked: NoneType"
        ]

        # When
        result = print_dict_types(input_data)

        # Then
        assert result == expected

    def test_dictionary_with_tuple(self):
        """Test with dictionary containing a tuple"""
        # Given
        input_data = {
            "coords": (1.0, 2.0, {"z": 3}),
            "name": "point"
        }
        expected = [
            "coords: tuple",
            "  coords[0]: float",
            "  coords[1]: float",
            "  coords[2]: dict",
            "    coords[2].z: int",
            "name: str"
        ]

        # When
        result = print_dict_types(input_data)

        # Then
        assert result == expected
