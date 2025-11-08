import pytest
from jet.scrapers.preprocessor import split_markdown_by_headings


@pytest.fixture
def sample_markdown() -> str:
    return """
# Alpha
Intro text.

## Beta
Feature list.

### Gamma
Details here.
"""


def test_split_by_headings(sample_markdown):
    # Given
    result = split_markdown_by_headings(sample_markdown)

    # When
    expected = [
        "# Alpha\nIntro text.",
        "## Beta\nFeature list.",
        "### Gamma\nDetails here."
    ]

    # Then
    assert result == expected


def test_empty_text_under_header():
    # Given
    markdown = """
# Header 1

## Header 2
"""
    # When
    result = split_markdown_by_headings(markdown)

    # Then
    expected = [
        "# Header 1",
        "## Header 2"
    ]
    assert result == expected


def test_consecutive_headers_no_content():
    # Given
    markdown = """
# One
## Two
### Three
"""
    # When
    result = split_markdown_by_headings(markdown)

    # Then
    expected = [
        "# One",
        "## Two",
        "### Three",
    ]
    assert result == expected


def test_varying_heading_levels():
    # Given
    markdown = """
# Main
Top overview.

## Sub
Some details.

#### Deep
Nested details here.

# New Section
Back to top level.

###### Ultra Deep
Edge-case deepest header.

##### Slightly Shallower
Still deep but not deepest.

# Final
Closing remarks.
"""

    # When
    result = split_markdown_by_headings(markdown)

    # Then
    expected = [
        "# Main\nTop overview.",
        "## Sub\nSome details.",
        "#### Deep\nNested details here.",
        "# New Section\nBack to top level.",
        "###### Ultra Deep\nEdge-case deepest header.",
        "##### Slightly Shallower\nStill deep but not deepest.",
        "# Final\nClosing remarks.",
    ]

    assert result == expected


def test_no_headers_returns_full_text():
    # Given
    markdown = """
Just plain text.
No headers here.
"""
    # When
    result = split_markdown_by_headings(markdown)

    # Then
    expected = []
    assert result == expected


def test_trailing_and_leading_blank_lines():
    # Given
    markdown = """

# A
Alpha text.

## B

Beta content.

### C
Gamma text.

Gamma second text.


"""
    # When
    result = split_markdown_by_headings(markdown)

    # Then
    expected = [
        "# A\nAlpha text.",
        "## B\nBeta content.",
        "### C\nGamma text.\n\nGamma second text.",
    ]
    assert result == expected
