# tests/test_extraction.py
from typing import Any, List
import pytest

from page_utils import (
    ExtractedElement,
    extract_referenced_elements,
    _parse_tag_key,
)


@pytest.fixture
def empty_result() -> List[ExtractedElement]:
    return []


# =============================================================================
# Simple flat cases
# =============================================================================

def test_extracts_single_ref_in_flat_dict():
    # Given
    data = {
        "generic [ref=e01]": "Hello world"
    }

    # When
    result = extract_referenced_elements(data)

    # Then
    assert len(result) == 1
    el = result[0]
    assert el["ref"] == "e01"
    assert el["kind"] == "div"
    assert el["text"] == "Hello world"
    assert el["is_leaf"] is True
    assert el["children_count"] == 0


def test_extracts_link_with_quoted_text_and_url():
    # Given
    data = {
        'link "Contact Us" [ref=abc123] [cursor=pointer]': [
            {"/url": "https://example.com/contact"},
            "img [ref=img1]"
        ]
    }

    # When
    result = extract_referenced_elements(data)

    # Then
    assert len(result) == 2  # link + img
    link = next(el for el in result if el["ref"] == "abc123")
    assert link["kind"] == "link"
    assert link["text"] == "Contact Us"
    assert link["url"] == "https://example.com/contact"
    assert link["attributes"] == {"cursor": "pointer"}
    assert link["children_count"] == 2
    assert link["is_leaf"] is False


# =============================================================================
# Nested structures
# =============================================================================

def test_handles_deeply_nested_structure():
    # Given
    data = {
        "main [ref=m1]": [
            {
                "navigation [ref=nav1]": [
                    {
                        'link "Home" [ref=l1]': [
                            {"/url": "/"},
                            {"generic [ref=g1]": "Home icon"}
                        ]
                    }
                ]
            },
            "footer [ref=f1]"
        ]
    }

    # When
    result = extract_referenced_elements(data)

    # Then
    refs = {el["ref"] for el in result}
    assert refs == {"m1", "nav1", "l1", "g1", "f1"}

    home_link = next(el for el in result if el["ref"] == "l1")
    assert home_link["text"] == "Home"
    assert home_link["url"] == "/"
    assert home_link["children_count"] == 2


# =============================================================================
# Multiple attributes + active flag (no value)
# =============================================================================

def test_parses_attribute_without_value():
    # Given
    data = {
        "button [active] [ref=btn-submit] [type=submit]": "Submit"
    }

    # When
    result = extract_referenced_elements(data)

    # Then
    el = result[0]
    assert el["ref"] == "btn-submit"
    assert el["kind"] == "button"
    assert el["attributes"] == {"active": "true", "type": "submit"}
    assert el["text"] == "Submit"


# =============================================================================
# Edge cases
# =============================================================================

@pytest.mark.parametrize("bad_input", [
    None,
    123,
    "just string",
    ["list", "of", "strings"],
])
def test_handles_invalid_root_types_gracefully(bad_input: Any):
    # When
    result = extract_referenced_elements(bad_input)

    # Then
    assert result == []  # should not crash


def test_no_ref_elements_are_ignored():
    # Given
    data = {
        "generic": "plain text",
        "div": {"class": "container"},
        "p": "paragraph without ref"
    }

    # When
    result = extract_referenced_elements(data)

    # Then
    assert len(result) == 0


def test_multiple_refs_in_nested_list():
    # Given
    data = [
        {"generic [ref=a1]": "first"},
        {
            "section [ref=s1]": [
                {"article [ref=art1]": "content"},
                {"generic [ref=a2]": "second"}
            ]
        }
    ]

    # When
    result = extract_referenced_elements(data)

    # Then
    refs = sorted([el["ref"] for el in result])
    assert refs == ["a1", "a2", "art1", "s1"]


class TestParseTagKey:
    """Given a tag key string
    When we parse it with _parse_tag_key
    Then we should get correct tag, text, ref and attributes
    """

    @pytest.mark.parametrize(
        "key, expected_tag, expected_text, expected_ref, expected_attrs",
        [
            # Case 1: simple generic with ref
            (
                "generic [ref=e1]",
                "generic",
                None,
                "e1",
                {},
            ),
            # Case 2: link with quoted text + attributes
            (
                'link "Contact Us" [ref=abc123] [cursor=pointer]',
                "link",
                "Contact Us",
                "abc123",
                {"cursor": "pointer"},
            ),
            # Case 3: button with flag attribute + ref
            (
                "button [active] [ref=btn-submit] [type=submit]",
                "button",
                None,
                "btn-submit",
                {"active": "true", "type": "submit"},
            ),
            # Case 4: combobox with long quoted text
            (
                'combobox "Search language" [ref=e76]',
                "combobox",
                "Search language",
                "e76",
                {},
            ),
            # Case 5: no ref → should return None for ref
            (
                'generic "Plain text"',
                "generic",
                "Plain text",
                None,
                {},
            ),
            # Case 6: ref appears before other attributes
            (
                'link [ref=l99] [role=navigation] "Home"',
                "link",
                "Home",
                "l99",
                {"role": "navigation"},
            ),
        ]
    )
    def test_parsing_various_formats(
        self,
        key: str,
        expected_tag: str,
        expected_text: str | None,
        expected_ref: str | None,
        expected_attrs: dict[str, str],
    ):
        # When
        tag, text, ref, attrs = _parse_tag_key(key)

        # Then
        assert tag == expected_tag, f"Tag mismatch for {key!r}"
        assert text == expected_text, f"Text mismatch for {key!r}"
        assert ref == expected_ref, f"Ref mismatch for {key!r}"
        assert attrs == expected_attrs, f"Attrs mismatch for {key!r}"
