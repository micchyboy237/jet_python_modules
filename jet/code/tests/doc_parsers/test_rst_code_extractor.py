import pytest
import os
from jet.code.rst_code_extractor import rst_to_code_blocks
from typing import List, Dict


@pytest.fixture
def temp_rst_file(tmp_path):
    """Create a temporary RST file for testing."""
    rst_content = """
Regular Code Block
================

.. code-block:: python

    print("Hello, World!")

Doctest Block
=============

.. doctest::

    >>> from textblob import TextBlob
    >>> blob = TextBlob("Hello world")
    >>> blob.words
    WordList(['Hello', 'world'])

Plain Text Block
===============

.. code-block:: text

    This is a plain text block
"""
    rst_file = tmp_path / "test.rst"
    rst_file.write_text(rst_content)
    return str(rst_file)


@pytest.fixture
def cleanup():
    """Ensure clean up after tests."""
    yield
    # No specific cleanup needed for these tests


class TestRstCodeExtractor:
    """Test suite for rst_to_code_blocks function."""

    def test_extract_python_code_block(self, temp_rst_file: str, cleanup):
        """Test extraction of a regular Python code block."""
        # Given: An RST file with a Python code block
        expected: List[Dict[str, str]] = [
            {'type': 'python', 'code': 'print("Hello, World!")'},
            {'type': 'python', 'code': '>>> from textblob import TextBlob\n>>> blob = TextBlob("Hello world")\n>>> blob.words\nWordList(['Hello', 'world'])'},
            {'type': 'text', 'code': 'This is a plain text block'},
        ]

        # When: We extract code blocks
        result = rst_to_code_blocks(temp_rst_file)

        # Then: The extracted code blocks should match expected
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_extract_doctest_block(self, temp_rst_file: str, cleanup):
        """Test extraction of a doctest block."""
        # Given: An RST file with a doctest block
        expected: list[dict[str, str]] = [
            {'type': 'python',
                'code': ">>> from textblob import TextBlob\n>>> blob = TextBlob(\"Hello world\")\n>>> blob.words\nWordList(['Hello', 'world'])"}
        ]

        # When: We extract code blocks and filter for doctest
        result = [block for block in rst_to_code_blocks(
            temp_rst_file) if block['code'].startswith('>>>')]

        # Then: The extracted doctest block should match expected
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_extract_text_block(self, temp_rst_file: str, cleanup):
        """Test extraction of a text code block."""
        # Given: An RST file with a text code block
        expected: List[Dict[str, str]] = [
            {'type': 'text', 'code': 'This is a plain text block'}
        ]

        # When: We extract code blocks and filter for text type
        result = [block for block in rst_to_code_blocks(
            temp_rst_file) if block['type'] == 'text']

        # Then: The extracted text block should match expected
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_empty_rst_file(self, tmp_path: str, cleanup):
        """Test handling of an empty RST file."""
        # Given: An empty RST file
        empty_rst = tmp_path / "empty.rst"
        empty_rst.write_text("")
        expected: List[Dict[str, str]] = []

        # When: We extract code blocks from empty file
        result = rst_to_code_blocks(str(empty_rst))

        # Then: No code blocks should be returned
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_nonexistent_file(self, cleanup):
        """Test handling of a nonexistent RST file."""
        # Given: A nonexistent file path
        nonexistent_path = "/nonexistent/path/test.rst"

        # When: We try to extract code blocks
        # Then: FileNotFoundError should be raised
        with pytest.raises(FileNotFoundError):
            rst_to_code_blocks(nonexistent_path)
