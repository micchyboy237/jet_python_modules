import json
import os
import tempfile
import unittest
from pathlib import Path
from jet.code.extraction.extract_comments import extract_text_from_ipynb, extract_comments


class TestExtractComments(unittest.TestCase):
    def setUp(self):
        """Set up temporary directory and files for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def create_notebook(self, filename, cells):
        """Helper to create a temporary notebook file."""
        nb = {"cells": cells, "metadata": {},
              "nbformat": 4, "nbformat_minor": 5}
        file_path = self.temp_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f)
        return file_path

    def test_extract_text_from_ipynb_markdown(self):
        """Test extracting markdown content from a notebook."""
        cells = [
            {"cell_type": "markdown", "source": ["# Title\n", "Some text"]},
            {"cell_type": "code", "source": ["print('hello')"], "outputs": []}
        ]
        nb_path = self.create_notebook("test.ipynb", cells)
        result = extract_text_from_ipynb(
            nb_path, include_outputs=False, include_code=False, include_comments=False)
        expected = "# Title\nSome text\n"
        self.assertEqual(result, expected)

    def test_extract_text_from_ipynb_code_with_comments(self):
        """Test extracting code with comments when include_comments=True."""
        cells = [
            {"cell_type": "code", "source": [
                "# Comment\n", "print('hello')\n", '"""Docstring"""']},
            {"cell_type": "markdown", "source": ["Some markdown"]}
        ]
        nb_path = self.create_notebook("test.ipynb", cells)
        result = extract_text_from_ipynb(
            nb_path, include_outputs=False, include_code=True, include_comments=True)
        expected = "```python\n# Comment\nprint('hello')\n\"\"\"Docstring\"\"\"\n```\n\nSome markdown\n"
        self.assertEqual(result, expected)

    def test_extract_text_from_ipynb_outputs(self):
        """Test extracting outputs when include_outputs=True."""
        cells = [
            {"cell_type": "code", "source": ["print('hello')"], "outputs": [
                {"output_type": "stream", "text": ["hello\n"]}]}
        ]
        nb_path = self.create_notebook("test.ipynb", cells)
        result = extract_text_from_ipynb(
            nb_path, include_outputs=True, include_code=False, include_comments=False)
        expected = "```output\nhello\n```\n"
        self.assertEqual(result, expected)

    def test_extract_text_from_ipynb_invalid_file(self):
        """Test handling of invalid notebook file."""
        file_path = self.temp_path / "invalid.ipynb"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("not a json")
        result = extract_text_from_ipynb(
            file_path, include_outputs=False, include_code=False, include_comments=False)
        self.assertIsNone(result)

    def test_extract_comments_single_file(self):
        """Test extract_comments with a single notebook file."""
        cells = [{"cell_type": "markdown", "source": ["# Title\n", "Text"]}]
        nb_path = self.create_notebook("test.ipynb", cells)
        result = extract_comments(
            nb_path, include_outputs=False, include_code=False, include_comments=False)
        expected = "# Title\nText\n"
        self.assertEqual(result, expected)

    def test_extract_comments_directory(self):
        """Test extract_comments with a directory of notebooks."""
        cells1 = [{"cell_type": "markdown", "source": ["# Notebook 1"]}]
        cells2 = [{"cell_type": "markdown", "source": ["# Notebook 2"]}]
        self.create_notebook("nb1.ipynb", cells1)
        self.create_notebook("nb2.ipynb", cells2)
        result = extract_comments(
            self.temp_path, include_outputs=False, include_code=False, include_comments=False)
        expected = "\n# nb1.ipynb\n# Notebook 1\n\n\n\n# nb2.ipynb\n# Notebook 2\n\n\n"
        self.assertEqual(result, expected)

    def test_extract_comments_handles_none(self):
        """Test extract_comments handles None return from extract_text_from_ipynb."""
        file_path = self.temp_path / "invalid.ipynb"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("not a json")
        result = extract_comments(
            file_path, include_outputs=False, include_code=False, include_comments=False)
        # Should return empty string, skipping invalid file
        self.assertEqual(result, "")

    def test_extract_comments_empty_directory(self):
        """Test extract_comments with an empty directory."""
        result = extract_comments(
            self.temp_path, include_outputs=False, include_code=False, include_comments=False)
        self.assertEqual(result, "")


if __name__ == '__main__':
    unittest.main()
