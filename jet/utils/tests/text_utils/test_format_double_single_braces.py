import pytest

from jet.utils.text import format_double_single_braces


@pytest.mark.parametrize(
    "given_text,expected",
    [
        # Given: basic isolated braces
        # When: function is called
        # Then: each single brace becomes double
        pytest.param("a { b } c", "a {{ b }} c", id="basic singles"),

        # Given: already doubled braces
        # When: function is called
        # Then: no change
        pytest.param("{{already}} doubled", "{{already}} doubled", id="already doubled"),

        # Given: mix of single and double braces
        # When: function is called
        # Then: only singles are doubled
        pytest.param("{single} {mix} {{double}}", "{{single}} {{mix}} {{double}}", id="mixed"),

        # Given: no braces
        # When: function is called
        # Then: output equals input
        pytest.param("no braces here", "no braces here", id="no braces"),

        # Given: adjacent different braces
        # When: function is called
        # Then: both are single and isolated → both doubled
        pytest.param("{}", "{{}}", id="adjacent different"),

        # Given: empty string
        # When: function is called
        # Then: returns empty string
        pytest.param("", "", id="empty string"),

        # Given: triple consecutive braces
        # When: function is called
        # Then: all are part of a run → no change
        pytest.param("{{{}}}", "{{{}}}", id="triple braces remain unchanged"),

        # Given: odd-length run of same brace
        # When: function is called
        # Then: middle one is not isolated → no doubling
        pytest.param("{{{", "{{{", id="three opening - unchanged"),
        pytest.param("}}}", "}}}", id="three closing - unchanged"),

        # Given: single brace surrounded by text
        # When: function is called
        # Then: doubled
        pytest.param("before{after", "before{{after", id="single open mid-word"),
        pytest.param("before}after", "before}}after", id="single close mid-word"),
    ],
)

def test_double_single_braces(given_text: str, expected: str) -> None:
    # Given: input text as provided
    input_text: str = given_text

    # When: calling the function
    result: str = format_double_single_braces(input_text)

    # Then: result must exactly match expected output
    assert result == expected
