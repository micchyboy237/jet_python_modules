from unittest.mock import patch
from jet.utils.file import matches_pattern


class TestMatchesPattern:
    """Test the matches_pattern function for various path and pattern matching scenarios."""

    def test_exact_path_match(self):
        """Given an exact path pattern, should return True for matching full path."""
        # Given
        path = "/path/to/docs/file1.ipynb"
        patterns = ["/path/to/docs/file1.ipynb"]
        expected = True
        
        # When
        result = matches_pattern(path, patterns)
        
        # Then
        assert result == expected

    def test_exact_filename_match(self):
        """Given an exact filename pattern, should match regardless of directory."""
        # Given
        path = "/different/path/file1.ipynb"
        patterns = ["file1.ipynb"]
        expected = True
        
        # When
        result = matches_pattern(path, patterns)
        
        # Then
        assert result == expected

    def test_wildcard_filename_match(self):
        """Given a wildcard filename pattern, should match files with matching names."""
        # Given
        path = "/path/to/docs/file1.ipynb"
        patterns = ["*.ipynb"]
        expected = True
        
        # When
        result = matches_pattern(path, patterns)
        
        # Then
        assert result == expected

    def test_directory_component_exact_match(self):
        """Given an exact directory name pattern, should match files in that directory."""
        # Given
        path = "/path/to/docs/file1.ipynb"
        patterns = ["docs"]
        expected = True
        
        # When
        result = matches_pattern(path, patterns)
        
        # Then
        assert result == expected

    def test_directory_component_wildcard_match(self):
        """Given a wildcard directory pattern, should match files in matching directories."""
        # Given
        path = "/path/to/docs/subfolder/file2.ipynb"
        patterns = ["doc*"]
        expected = True
        
        # When
        result = matches_pattern(path, patterns)
        
        # Then
        assert result == expected

    def test_partial_path_wildcard_match(self):
        """Given a partial path with wildcard, should match containing paths."""
        # Given
        path = "/project/docs/analysis/file1.ipynb"
        patterns = ["*docs*"]
        expected = True
        
        # When
        result = matches_pattern(path, patterns)
        
        # Then
        assert result == expected

    def test_path_with_separator_pattern(self):
        """Given a pattern with directory separator, should match exact directory structure."""
        # Given
        path = "/path/to/docs/file1.ipynb"
        patterns = ["to/docs"]
        expected = True
        
        # When
        result = matches_pattern(path, patterns)
        
        # Then
        assert result == expected

    def test_no_match_empty_patterns(self):
        """Given empty patterns list, should return False."""
        # Given
        path = "/path/to/file.txt"
        patterns = []
        expected = False
        
        # When
        result = matches_pattern(path, patterns)
        
        # Then
        assert result == expected

    def test_no_match_unrelated_pattern(self):
        """Given patterns that don't match path, filename, or directories, should return False."""
        # Given
        path = "/path/to/docs/file1.ipynb"
        patterns = ["unrelated.txt", "other_dir"]
        expected = False
        
        # When
        result = matches_pattern(path, patterns)
        
        # Then
        assert result == expected

    def test_multiple_patterns_first_match(self):
        """Given multiple patterns where first matches, should return True immediately."""
        # Given
        path = "/path/to/docs/file1.ipynb"
        patterns = ["docs", "unrelated"]
        expected = True
        
        # When
        result = matches_pattern(path, patterns)
        
        # Then
        assert result == expected

    def test_multiple_patterns_last_match(self):
        """Given multiple patterns where only last matches, should return True."""
        # Given
        path = "/path/to/docs/file1.ipynb"
        patterns = ["unrelated", "*.ipynb"]
        expected = True
        
        # When
        result = matches_pattern(path, patterns)
        
        # Then
        assert result == expected

    def test_windows_path_separators(self):
        """Given Windows-style paths with backslashes, should handle normalization correctly."""
        # Given
        path = r"C:\project\docs\file1.ipynb"
        patterns = [r"docs"]
        expected = True
        
        # When
        result = matches_pattern(path, patterns)
        
        # Then
        assert result == expected

    @patch('os.path.normpath')
    def test_normalized_path_handling(self, mock_normpath):
        """Given mocked normpath behavior, should use normalized paths for matching."""
        # Given
        mock_normpath.side_effect = lambda x: x.replace('\\', '/')
        path = r"C:\path\to\docs\file1.ipynb"
        patterns = ["to/docs"]
        expected = True
        
        # When
        result = matches_pattern(path, patterns)
        
        # Then
        assert result == expected
        mock_normpath.assert_called()

    def test_relative_path_matching(self):
        """Given relative paths, should match patterns correctly."""
        # Given
        path = "docs/subfolder/file2.ipynb"
        patterns = ["subfolder"]
        expected = True
        
        # When
        result = matches_pattern(path, patterns)
        
        # Then
        assert result == expected

    def test_deep_directory_structure(self):
        """Given deeply nested paths, should match any directory component."""
        # Given
        path = "/project/src/utils/helpers/file.py"
        patterns = ["helpers"]
        expected = True
        
        # When
        result = matches_pattern(path, patterns)
        
        # Then
        assert result == expected