# tests/test__flatten_list_groups.py
"""
Unit tests for the internal helper _flatten_list_groups from _markdown_analyzer.py
"""

from typing import Any

import pytest
from jet.code.markdown_types.base_markdown_analysis_types import ListItem
from jet.code.markdown_utils._markdown_analyzer import _flatten_list_groups


def item(text: str, checked: bool = False, task_item: bool = False) -> ListItem:
    """Quick factory for ListItem dicts"""
    return {
        "text": text,
        "checked": checked,
        "task_item": task_item,
    }


@pytest.mark.parametrize(
    "case_name, input_groups, expected_items",
    [
        # ────────────────────────────────────────────────
        (
            "empty input → empty output",
            [],
            [],
        ),
        # ────────────────────────────────────────────────
        (
            "single empty group → empty output",
            [[]],
            [],
        ),
        # ────────────────────────────────────────────────
        (
            "multiple empty groups → empty output",
            [[], [], []],
            [],
        ),
        # ────────────────────────────────────────────────
        (
            "one group with items → flat list with same items",
            [
                [
                    item("Buy milk"),
                    item("Buy bread"),
                ]
            ],
            [
                item("Buy milk"),
                item("Buy bread"),
            ],
        ),
        # ────────────────────────────────────────────────
        (
            "multiple groups → concatenated in order",
            [
                [item("Apples"), item("Bananas")],
                [],
                [item("Cherries"), item("Dates")],
                [item("Eggs")],
            ],
            [
                item("Apples"),
                item("Bananas"),
                item("Cherries"),
                item("Dates"),
                item("Eggs"),
            ],
        ),
        # ────────────────────────────────────────────────
        (
            "task items are preserved correctly",
            [
                [
                    item("Write report", task_item=True, checked=False),
                    item("Review PR", task_item=True, checked=True),
                ],
                [
                    item("Deploy to prod", task_item=True, checked=False),
                ],
            ],
            [
                item("Write report", task_item=True, checked=False),
                item("Review PR", task_item=True, checked=True),
                item("Deploy to prod", task_item=True, checked=False),
            ],
        ),
        # ────────────────────────────────────────────────
        (
            "non-list elements in groups are skipped",
            [
                [item("Valid 1"), item("Valid 2")],
                "this is not a list",  # type: str
                123,  # type: int
                [item("Valid 3")],
                {"not": "a list either"},
            ],
            [
                item("Valid 1"),
                item("Valid 2"),
                item("Valid 3"),
            ],
        ),
        # ────────────────────────────────────────────────
        (
            "groups with mixed valid/invalid items",
            [
                [
                    item("Coffee"),
                    None,  # invalid
                    item("Tea"),
                ],
                [42, item("Sugar")],  # mixed
            ],
            [
                item("Coffee"),
                item("Tea"),
                item("Sugar"),
            ],
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_flatten_list_groups_behaviors(
    case_name: str,
    input_groups: list[list[Any]],
    expected_items: list[ListItem],
):
    # Given
    groups = input_groups

    # When
    result = _flatten_list_groups(groups)

    # Then
    assert result == expected_items, f"Flattening failed for case: {case_name}"

    # Additional structural check
    for item_dict in result:
        assert isinstance(item_dict, dict)
        assert "text" in item_dict
        assert isinstance(item_dict["text"], str)
        assert "checked" in item_dict
        assert "task_item" in item_dict


def test_flatten_list_groups_returns_new_list():
    """Ensure we don't mutate input and return a fresh list"""
    original = [[item("A"), item("B")]]
    original_copy = [group[:] for group in original]  # deep enough for this test

    result = _flatten_list_groups(original)

    # Mutate result
    if result:
        result[0]["text"] = "Modified"

    # Original should be unchanged
    assert original == original_copy, "Input was mutated"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
