# tests/test_flatten_nested_content.py
"""
Unit tests for flatten_nested_content() helper used in base_parse_markdown
"""

import json

import pytest
from jet.code.markdown_utils._markdown_parser import flatten_nested_content


def test_flatten_nested_content_string_top_level():
    # Given
    content = "Simple paragraph\nwith two lines"
    expected = "Simple paragraph\nwith two lines"  # top-level keeps original newlines

    # When
    result = flatten_nested_content(content, depth=0)

    # Then
    assert result == expected


def test_flatten_nested_content_string_indented():
    # Given
    content = "Nested text\nSecond line"
    expected = "  - First level\n  Second line"

    # When
    result = flatten_nested_content(content, depth=1, prefix="- ", is_ordered=False)

    # Then
    assert result == expected


def test_flatten_nested_content_empty_cases():
    # Given + When + Then
    assert flatten_nested_content("") == ""
    assert flatten_nested_content("   \n  ") == ""
    assert flatten_nested_content([]) == ""
    assert flatten_nested_content({}) == "{}"


def test_flatten_nested_content_simple_list():
    # Given
    items = ["Apples", "Bananas", "Cherries"]
    expected = "- Apples\n- Bananas\n- Cherries"

    # When
    result = flatten_nested_content(items, depth=0, is_ordered=False)

    # Then
    assert result == expected


def test_flatten_nested_content_ordered_list():
    # Given
    items = ["Step one", "Step two", "Step three"]
    expected = "1. Step one\n2. Step two\n3. Step three"

    # When
    result = flatten_nested_content(items, depth=0, is_ordered=True)

    # Then
    assert result == expected


def test_flatten_nested_content_nested_lists():
    # Given - realistic nested markdown list structure
    nested = [
        "Fruits",
        [
            "Apples",
            "Bananas",
            ["Green", "Yellow", "Red"],
        ],
        "Vegetables",
        ["Carrot", "Potato"],
    ]

    expected = (
        "- Fruits\n"
        "  - Apples\n"
        "  - Bananas\n"
        "    - Green\n"
        "    - Yellow\n"
        "    - Red\n"
        "- Vegetables\n"
        "  - Carrot\n"
        "  - Potato"
    )

    # When
    result = flatten_nested_content(nested, depth=0, is_ordered=False)

    # Then
    assert result == expected


def test_flatten_nested_content_dict_with_text():
    # Given
    token_like = {"text": "This is the content", "other": "ignored"}
    expected = "This is the content"

    # When
    result = flatten_nested_content(token_like, depth=0)

    # Then
    assert result == expected


def test_flatten_nested_content_dict_with_items():
    # Given - simulates ListMeta-like structure
    structure = {
        "items": [
            {"text": "Level 1 A"},
            {"text": ["Sub A1", "Sub A2"]},
            {"text": "Level 1 B"},
        ],
        "ordered": True,
    }

    expected = "1. Level 1 A\n2. Sub A1\n   Sub A2\n3. Level 1 B"

    # When
    result = flatten_nested_content(structure, depth=0)

    # Then
    assert result == expected


def test_flatten_nested_content_table_cell_multi_line():
    # Given - typical table cell content after parsing
    cell_content = "First line\nSecond line\nThird"

    # When - depth=0 as used in table flattening
    result = (
        flatten_nested_content(cell_content, depth=0).strip().replace("\n", " <br> ")
    )

    # Then
    assert result == "First line <br> Second line <br> Third"


def test_flatten_nested_content_task_item_nested():
    # Given - nested content inside a task list item
    nested_task = {
        "text": [
            "Main task description",
            ["Sub-task 1", "Sub-task 2"],
        ]
    }

    expected = "- Main task description\n  - Sub-task 1\n  - Sub-task 2"

    # When
    result = flatten_nested_content(nested_task["text"], depth=0, prefix="- ")

    # Then
    assert result == expected


def test_flatten_nested_content_fallback_json():
    # Given - unknown complex structure
    weird = {"a": 1, "b": [2, 3], "c": {"d": 4}}

    # When
    result = flatten_nested_content(weird, depth=0)

    # Then
    expected_json = json.dumps(weird, ensure_ascii=False, indent=2)
    assert result == expected_json


@pytest.mark.parametrize(
    "input_content, expected",
    [
        (["A", "", "B"], "- A\n- B"),  # skips empty strings
        ([[], ["X"]], "- X"),  # skips empty sublists
        ({"text": ""}, ""),  # empty text field
    ],
)
def test_flatten_skips_empty_content(input_content, expected):
    result = flatten_nested_content(input_content, depth=0, is_ordered=False)
    assert result.strip() == expected.strip()
