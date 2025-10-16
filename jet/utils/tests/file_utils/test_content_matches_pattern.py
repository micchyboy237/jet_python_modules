from unittest.mock import patch, mock_open
from jet.utils.file import content_matches_pattern


class TestContentMatchesPattern:
    """Test the content_matches_pattern function for text pattern matching in files."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_exact_content_match(self, mock_exists, mock_open):
        """Given exact content pattern, should return True when content matches."""
        # Given
        file_path = "/path/to/file.txt"
        patterns = ["exact content here"]
        mock_exists.return_value = True
        mock_open.return_value.read.return_value = "exact content here"
        expected = True
        
        # When
        result = content_matches_pattern(file_path, patterns)
        
        # Then
        assert result == expected
        mock_open.assert_called_once_with(file_path, 'r', encoding='utf-8')

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_wildcard_content_match(self, mock_exists, mock_open):
        """Given wildcard content pattern, should return True when content matches wildcard."""
        # Given
        file_path = "/path/to/file.txt"
        patterns = ["*content*"]
        mock_exists.return_value = True
        mock_open.return_value.read.return_value = "some content here"
        expected = True
        
        # When
        result = content_matches_pattern(file_path, patterns)
        
        # Then
        assert result == expected

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_multiple_patterns_first_match(self, mock_exists, mock_open):
        """Given multiple patterns where first matches, should return True."""
        # Given
        file_path = "/path/to/file.txt"
        patterns = ["*content*", "no match"]
        mock_exists.return_value = True
        mock_open.return_value.read.return_value = "some content here"
        expected = True
        
        # When
        result = content_matches_pattern(file_path, patterns)
        
        # Then
        assert result == expected

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_no_content_match(self, mock_exists, mock_open):
        """Given patterns that don't match content, should return False."""
        # Given
        file_path = "/path/to/file.txt"
        patterns = ["missing pattern"]
        mock_exists.return_value = True
        mock_open.return_value.read.return_value = "actual content"
        expected = False
        
        # When
        result = content_matches_pattern(file_path, patterns)
        
        # Then
        assert result == expected

    @patch('os.path.exists')
    def test_file_not_found(self, mock_exists):
        """Given non-existent file, should return False."""
        # Given
        file_path = "/nonexistent/file.txt"
        patterns = ["*"]
        mock_exists.return_value = False
        expected = False
        
        # When
        result = content_matches_pattern(file_path, patterns)
        
        # Then
        assert result == expected

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_encoding_error(self, mock_exists, mock_open):
        """Given file with encoding issues, should return False."""
        # Given
        file_path = "/path/to/file.txt"
        patterns = ["*"]
        mock_exists.return_value = True
        mock_open.side_effect = UnicodeDecodeError('utf-8', b'', 0, 1, 'error')
        expected = False
        
        # When
        result = content_matches_pattern(file_path, patterns)
        
        # Then
        assert result == expected

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_empty_patterns(self, mock_exists, mock_open):
        """Given empty patterns list, should return False."""
        # Given
        file_path = "/path/to/file.txt"
        patterns = []
        mock_exists.return_value = True
        expected = False
        
        # When
        result = content_matches_pattern(file_path, patterns)
        
        # Then
        assert result == expected

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_empty_file_content(self, mock_exists, mock_open):
        """Given empty file content, should return False unless pattern matches empty string."""
        # Given
        file_path = "/path/to/empty.txt"
        patterns = ["non-empty"]
        mock_exists.return_value = True
        mock_open.return_value.read.return_value = ""
        expected = False
        
        # When
        result = content_matches_pattern(file_path, patterns)
        
        # Then
        assert result == expected