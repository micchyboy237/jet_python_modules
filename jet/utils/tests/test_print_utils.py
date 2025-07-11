import pytest
from typing import List, Any, Mapping
from jet.utils.print_utils import get_common_dict_structure, print_dict_types


class TestPrintDictTypes:
    def test_basic_dict_sorting(self):
        # Given: A dictionary with nested dictionaries of varying key counts
        input_data: Mapping[str, Any] = {
            "z": {"p": 1},
            "a": {"x": 1, "y": 2, "z": 3},
            "b": {"x": 1, "y": 2}
        }
        expected: List[str] = [
            "a: dict",
            "  a.x: int",
            "  a.y: int",
            "  a.z: int",
            "b: dict",
            "  b.x: int",
            "  b.y: int",
            "z: dict",
            "  z.p: int"
        ]

        # When: Processing the dictionary
        result = print_dict_types(input_data)

        # Then: Dictionaries should be sorted by key count in descending order
        assert result == expected, "Dictionaries should be sorted by key count"

    def test_list_of_dicts_sorting(self):
        # Given: A dictionary with a list of dictionaries
        input_data: Mapping[str, Any] = {
            "items": [
                {"a": 1},
                {"p": 1, "q": 2},
                {"x": 1, "y": 2, "z": 3}
            ]
        }
        expected: List[str] = [
            "items[]: list[dict]",
            "  items[].a: int",
            "  items[].p: int",
            "  items[].q: int",
            "  items[].x: int",
            "  items[].y: int",
            "  items[].z: int"
        ]

        # When: Processing the dictionary
        result = print_dict_types(input_data)

        # Then: List items should be sorted by key count
        assert result == expected, "List of dictionaries should be sorted by key count"

    def test_empty_dict_and_list(self):
        # Given: A dictionary with empty structures
        input_data: Mapping[str, Any] = {
            "empty_dict": {},
            "empty_list": [],
            "data": {"x": 1}
        }
        expected: List[str] = [
            "data: dict",
            "  data.x: int",
            "empty_dict: dict",
            "empty_list: list"
        ]

        # When: Processing the dictionary
        result = print_dict_types(input_data)

        # Then: Empty structures should be handled correctly
        assert result == expected, "Empty dict and list should be handled and sorted"

    def test_nested_list_of_dicts(self):
        # Given: A nested list of dictionaries
        input_data: Mapping[str, Any] = {
            "data": [
                {"a": {"x": 1, "y": 2}},
                {"b": {"p": 1}}
            ]
        }
        expected: List[str] = [
            "data[]: list[dict]",
            "  data[].a: dict",
            "    data[].a.x: int",
            "    data[].a.y: int",
            "  data[].b: dict",
            "    data[].b.p: int"
        ]

        # When: Processing the dictionary
        result = print_dict_types(input_data)

        # Then: Nested list of dictionaries should be sorted
        assert result == expected, "Nested list of dictionaries should be sorted by key count"

    def test_mixed_types(self):
        # Given: A dictionary with mixed types including tuples
        input_data: Mapping[str, Any] = {
            "a": (1, {"x": 1, "y": 2}),
            "b": {"p": 1, "q": 2, "r": 3},
            "c": [1, "text"]
        }
        expected: List[str] = [
            "b: dict",
            "  b.p: int",
            "  b.q: int",
            "  b.r: int",
            "a: tuple",
            "  a[0]: int",
            "  a[1]: dict",
            "    a[1].x: int",
            "    a[1].y: int",
            "c: list",
            "  c: int",
            "  c: str"
        ]

        # When: Processing the dictionary
        result = print_dict_types(input_data)

        # Then: Mixed types should be handled and sorted correctly
        assert result == expected, "Mixed types should be sorted by key count where applicable"


class TestPrintDictTypes2:
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
            "  user.details: dict",
            "    user.details.age: int",
            "    user.details.city: str",
            "  user.name: str"
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
            "  items: dict",
            "    items.key: int",
            "  items: int",
            "  items: str",
            "  items: bool",
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
            "data[]: list[dict]",
            "  data[].id: int",
            "  data[].values: list",
            "    data[].values: int",
            "    data[].values: int"
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
            "  raw_tokens_sequential[].checked: NoneType",
            "  raw_tokens_sequential[].content: str",
            "  raw_tokens_sequential[].id: int",
            "  raw_tokens_sequential[].type: str"
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


class TestGetCommonDictStructure:
    def test_common_structure_valid_dicts(self):
        # Given: A list of dictionaries with various keys
        input_data: List[Mapping[str, Any]] = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 4, "b": 5, "d": 6},
            {"a": 7, "b": 8}
        ]
        expected: dict = {"a": 1, "b": 2, "c": 3, "d": 6}

        # When: Extracting structure with all possible keys
        result = get_common_dict_structure(input_data)

        # Then: The result should contain all keys with values from first occurrence
        assert result == expected, "Should return all keys with values from first occurrence"

    def test_empty_list(self):
        # Given: An empty list
        input_data: List[Mapping[str, Any]] = []
        expected: None = None

        # When: Extracting structure from empty list
        result = get_common_dict_structure(input_data)

        # Then: The result should be None
        assert result == expected, "Empty list should return None"

    def test_non_dict_list(self):
        # Given: A list with non-dictionary items
        input_data: List[Any] = [{"a": 1}, [1, 2], {"b": 2}]
        expected: None = None

        # When: Extracting structure from mixed list
        result = get_common_dict_structure(input_data)

        # Then: The result should be None
        assert result == expected, "Non-dictionary items should return None"

    def test_no_keys_in_first_dict(self):
        # Given: A list where the first dictionary is empty
        input_data: List[Mapping[str, Any]] = [
            {},
            {"a": 1, "b": 2},
            {"c": 3}
        ]
        expected: dict = {"a": 1, "b": 2, "c": 3}

        # When: Extracting structure
        result = get_common_dict_structure(input_data)

        # Then: The result should include all keys from subsequent dicts
        assert result == expected, "Empty first dict should include keys from others"

    def test_single_dict(self):
        # Given: A list with a single dictionary
        input_data: List[Mapping[str, Any]] = [{"a": 1, "b": 2}]
        expected: dict = {"a": 1, "b": 2}

        # When: Extracting structure
        result = get_common_dict_structure(input_data)

        # Then: The result should be the single dictionary
        assert result == expected, "Single dictionary should return itself"

    def test_all_possible_keys(self):
        # Given: A list of dictionaries with disjoint keys
        input_data: List[Mapping[str, Any]] = [
            {"a": 1, "b": 2},
            {"c": 3, "d": 4},
            {"e": 5}
        ]
        expected: dict = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}

        # When: Extracting structure with all possible keys
        result = get_common_dict_structure(input_data)

        # Then: The result should include all keys from all dicts
        assert result == expected, "Should include all keys from all dicts"
