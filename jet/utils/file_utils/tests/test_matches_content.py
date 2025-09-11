import os
import pytest
import tempfile
from jet.utils.file_utils.search import matches_content


class TestMatchesContent:
    def setup_method(self):
        """Create a temporary file for tests."""
        self.temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix='.txt').name
        with open(self.temp_file, 'w') as f:
            f.write("Hello, world! This is a test.")

    def teardown_method(self):
        """Clean up temporary file."""
        os.remove(self.temp_file)

    def test_matches_content_with_include_pattern(self):
        """Test content matching with an include pattern."""
        # Given: A file with specific content and an include pattern
        include_patterns = ["Hello*"]
        exclude_patterns = []
        expected = True

        # When: We check if the file content matches the include pattern
        result = matches_content(
            self.temp_file, include_patterns, exclude_patterns)

        # Then: The content should match the include pattern
        assert result == expected

    def test_matches_content_with_exclude_pattern(self):
        """Test content matching with an exclude pattern."""
        # Given: A file with specific content and an exclude pattern
        include_patterns = []
        exclude_patterns = ["world"]
        expected = False

        # When: We check if the file content does not match the exclude pattern
        result = matches_content(
            self.temp_file, include_patterns, exclude_patterns)

        # Then: The content should not match due to the exclude pattern
        assert result == expected

    def test_matches_content_case_insensitive(self):
        """Test case-insensitive content matching."""
        # Given: A file with content and a case-insensitive include pattern
        include_patterns = ["HELLO*"]
        exclude_patterns = []
        expected = True

        # When: We check content with case-insensitive flag
        result = matches_content(
            self.temp_file, include_patterns, exclude_patterns, case_sensitive=False)

        # Then: The content should match case-insensitively
        assert result == expected

    def test_matches_content_no_patterns(self):
        """Test content matching with no patterns."""
        # Given: A file with content and no patterns
        include_patterns = []
        exclude_patterns = []
        expected = True

        # When: We check content with no patterns
        result = matches_content(
            self.temp_file, include_patterns, exclude_patterns)

        # Then: The function should return True
        assert result == expected

    def test_matches_content_file_not_found(self):
        """Test content matching with a non-existent file."""
        # Given: A non-existent file path
        file_path = "/non/existent/file.txt"
        include_patterns = ["Hello*"]
        exclude_patterns = []
        expected = False

        # When: We attempt to check content of a non-existent file
        result = matches_content(file_path, include_patterns, exclude_patterns)

        # Then: The function should return False due to file error
        assert result == expected
