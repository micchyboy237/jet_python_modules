import unittest
from jet.file.validation import (
    has_format_placeholders,
    get_placeholders,
    format_with_placeholders,
    validate_match,
)


class TestValidation(unittest.TestCase):

    def test_has_format_placeholders_valid(self):
        """Test that verifies the presence of format placeholders."""
        self.assertTrue(has_format_placeholders("Hello, {}!"))

    def test_has_format_placeholders_escaped(self):
        """Test that checks escaped format placeholders."""
        self.assertFalse(has_format_placeholders(
            "This is a test with escaped \\{placeholder\\}"))

    def test_has_format_placeholders_escaped_placeholder(self):
        """Test that checks escaped placeholder inside text."""
        self.assertFalse(has_format_placeholders(
            "This is a \\{escaped placeholder\\} test."))

    def test_has_format_placeholders_non_escaped(self):
        """Test that checks non-escaped placeholders."""
        self.assertTrue(has_format_placeholders(
            "This is a \\{non-escaped\\} {placeholder}."))

    def test_get_placeholders(self):
        """Test extraction of placeholders from the text."""
        text = "Hello, {name}! Welcome to {place}."
        self.assertEqual(get_placeholders(text), ['name', 'place'])

    def test_format_placeholders_valid(self):
        """Test formatting placeholders with valid arguments."""
        text = "Hello, {name}! Welcome to {place}."
        formatted_text = format_with_placeholders(
            text, name="John", place="Paris")
        self.assertEqual(formatted_text, "Hello, John! Welcome to Paris.")

    def test_format_placeholders_missing_argument(self):
        """Test formatting placeholders when missing an argument."""
        text = "Hello, {name}! Welcome to {place}."
        with self.assertRaises(KeyError):
            format_with_placeholders(text, name="John")

    def test_validate_match_include_placeholder(self):
        """Test validation with include patterns using placeholders."""
        path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/.venv"
        include_patterns = ["{base}/bin/activate"]
        exclude_patterns = []
        self.assertTrue(validate_match(
            path, include_patterns, exclude_patterns))

    def test_validate_match_include_folder_placeholder(self):
        """Test validation with include patterns using folder placeholders."""
        path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/.venv"
        include_patterns = ["<folder>/bin/activate"]
        exclude_patterns = []
        self.assertTrue(validate_match(
            path, include_patterns, exclude_patterns))

    def test_validate_match_include_double_wildcard(self):
        """Test validation with double wildcard patterns."""
        path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/.venv"
        include_patterns = ["**/.venv"]
        exclude_patterns = []
        self.assertTrue(validate_match(
            path, include_patterns, exclude_patterns))

    def test_validate_match_include_path(self):
        """Test validation with a simple include pattern."""
        path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/.venv"
        include_patterns = ["**/JetScripts"]
        exclude_patterns = []
        self.assertTrue(validate_match(
            path, include_patterns, exclude_patterns))

    def test_validate_match_include_simple_folder(self):
        """Test validation with a simple folder pattern."""
        path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/.venv"
        include_patterns = ["**/JetScripts/*"]
        exclude_patterns = []
        self.assertTrue(validate_match(
            path, include_patterns, exclude_patterns))

    def test_validate_match_exclude_test_folder(self):
        """Test validation with exclude patterns matching a test folder."""
        path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/.venv"
        include_patterns = ["**/JetScripts/**"]
        exclude_patterns = ["**/test"]
        self.assertFalse(validate_match(
            path, include_patterns, exclude_patterns))

    def test_validate_match_exclude_test_file(self):
        """Test validation with exclude patterns matching a test file."""
        path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/.venv"
        include_patterns = ["**/JetScripts/*"]
        exclude_patterns = ["**/test/*"]
        self.assertFalse(validate_match(
            path, include_patterns, exclude_patterns))

    def test_validate_match_simple_folder(self):
        """Test validation with a simple folder pattern (e.g., .venv)."""
        path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/.venv"
        include_patterns = [".venv"]
        exclude_patterns = []
        self.assertTrue(validate_match(
            path, include_patterns, exclude_patterns))


if __name__ == "__main__":
    unittest.main()
