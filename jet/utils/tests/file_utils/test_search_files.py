import pytest
import os
from unittest.mock import patch, Mock

from jet.utils.file import search_files

@pytest.fixture
def setup_logger():
    logger = Mock()
    logger.info = Mock()
    logger.error = Mock()
    logger.warning = Mock()
    logger.log = Mock()
    logger.newline = Mock()
    logger.success = Mock()
    return logger

class TestSearchFiles:
    @pytest.fixture
    def mock_os_walk(self):
        def _walk(dir_path):
            # Normalize dir_path to avoid abspath issues
            normalized_path = os.path.normpath(dir_path)
            if normalized_path.endswith("path/to"):
                return [
                    ("/path/to/docs", ["subfolder"], ["file1.ipynb"]),
                    ("/path/to/docs/subfolder", [], ["file2.ipynb"]),
                    ("/path/to/tests", [], ["test_file.ipynb"]),
                    ("/path/to/scripts", [], ["script.py"]),
                ]
            return []
        return _walk

    def test_search_files_include_file_exact(self, setup_logger, mock_os_walk):
        """Given an exact include file name, only that file should be returned."""
        # Given
        base_dir = "/path/to"
        extensions = [".ipynb"]
        includes = ["file1.ipynb"]
        excludes = []
        expected_files = ["/path/to/docs/file1.ipynb"]

        # When
        with patch("os.walk", mock_os_walk):
            result = search_files(
                base_dir,
                extensions,
                include_files=includes,
                exclude_files=excludes,
            )

        # Then
        assert sorted(result) == sorted(expected_files)

    def test_search_files_include_file_wildcard(self, setup_logger, mock_os_walk):
        """Given a wildcard include file pattern, matching files should be returned."""
        # Given
        base_dir = "/path/to"
        extensions = [".ipynb"]
        includes = ["*.ipynb"]
        excludes = []
        expected_files = [
            "/path/to/docs/file1.ipynb",
            "/path/to/docs/subfolder/file2.ipynb",
            "/path/to/tests/test_file.ipynb",
        ]

        # When
        with patch("os.walk", mock_os_walk):
            result = search_files(
                base_dir,
                extensions,
                include_files=includes,
                exclude_files=excludes,
            )

        # Then
        assert sorted(result) == sorted(expected_files)

    def test_search_files_include_dir_exact(self, setup_logger, mock_os_walk):
        """Given an exact include directory name, files in that directory should be returned."""
        # Given
        base_dir = "/path/to"
        extensions = [".ipynb"]
        includes = ["docs"]
        excludes = []
        expected_files = [
            "/path/to/docs/file1.ipynb",
            "/path/to/docs/subfolder/file2.ipynb",
        ]

        # When
        with patch("os.walk", mock_os_walk):
            result = search_files(
                base_dir,
                extensions,
                include_files=includes,
                exclude_files=excludes,
            )

        # Then
        assert sorted(result) == sorted(expected_files)

    def test_search_files_include_dir_wildcard(self, setup_logger, mock_os_walk):
        """Given a wildcard include directory pattern, files in matching directories should be returned."""
        # Given
        base_dir = "/path/to"
        extensions = [".ipynb"]
        includes = ["doc*"]
        excludes = []
        expected_files = [
            "/path/to/docs/file1.ipynb",
            "/path/to/docs/subfolder/file2.ipynb",
        ]

        # When
        with patch("os.walk", mock_os_walk):
            result = search_files(
                base_dir,
                extensions,
                include_files=includes,
                exclude_files=excludes,
            )

        # Then
        assert sorted(result) == sorted(expected_files)

    def test_search_files_exclude_file_exact(self, setup_logger, mock_os_walk):
        """Given an exact exclude file name, that file should be excluded."""
        # Given
        base_dir = "/path/to"
        extensions = [".ipynb"]
        includes = []
        excludes = ["file1.ipynb"]
        expected_files = [
            "/path/to/docs/subfolder/file2.ipynb",
            "/path/to/tests/test_file.ipynb",
        ]

        # When
        with patch("os.walk", mock_os_walk):
            result = search_files(
                base_dir,
                extensions,
                include_files=includes,
                exclude_files=excludes,
            )

        # Then
        assert sorted(result) == sorted(expected_files)

    def test_search_files_exclude_dir_wildcard(self, setup_logger, mock_os_walk):
        """Given a wildcard exclude directory pattern, files in matching directories should be excluded."""
        # Given
        base_dir = "/path/to"
        extensions = [".ipynb"]
        includes = []
        excludes = ["*test*"]
        expected_files = [
            "/path/to/docs/file1.ipynb",
            "/path/to/docs/subfolder/file2.ipynb",
        ]

        # When
        with patch("os.walk", mock_os_walk):
            result = search_files(
                base_dir,
                extensions,
                include_files=includes,
                exclude_files=excludes,
            )

        # Then
        assert sorted(result) == sorted(expected_files)

    def test_search_files_combined_includes_excludes(self, setup_logger, mock_os_walk):
        """Given combined include and exclude patterns for files and directories, only matching files should be returned."""
        # Given
        base_dir = "/path/to"
        extensions = [".ipynb"]
        includes = ["docs", "*.ipynb"]
        excludes = ["subfolder", "test_file.ipynb"]
        expected_files = ["/path/to/docs/file1.ipynb"]

        # When
        with patch("os.walk", mock_os_walk):
            result = search_files(
                base_dir,
                extensions,
                include_files=includes,
                exclude_files=excludes,
            )

        # Then
        assert sorted(result) == sorted(expected_files)

class TestSearchFilesWithContentFilters:
    """Test search_files with content-based include/exclude filtering."""

    @pytest.fixture
    def mock_os_walk_txt(self):
        """Mock os.walk that returns .txt files for content filtering tests."""
        def _walk(dir_path):
            normalized_path = os.path.normpath(dir_path)
            if normalized_path.endswith("path/to"):
                return [
                    ("/path/to/docs", ["subfolder"], ["file1.txt", "other.txt"]),
                    ("/path/to/docs/subfolder", [], ["file2.txt"]),
                    ("/path/to/tests", [], ["test_file.txt"]),
                    ("/path/to/scripts", [], ["script.txt"]),
                ]
            return []
        return _walk

    def test_include_content_exact(self, setup_logger, mock_os_walk_txt):
        """Given exact content include pattern, only files with matching content should be returned."""
        # Given
        base_dir = "/path/to"
        extensions = [".txt"]
        includes = []
        excludes = []
        include_contents = ["exact content"]
        
        with patch('jet.utils.file.content_matches_pattern') as mock_content:
            # Mock to return True only for file1.txt (simulating content match)
            mock_content.side_effect = lambda path, pats: "file1.txt" in os.path.basename(path)
            expected_files = ["/path/to/docs/file1.txt"]
            
            with patch("os.walk", mock_os_walk_txt):
                result = search_files(
                    base_dir, extensions, 
                    include_files=includes, 
                    exclude_files=excludes,
                    include_contents=include_contents
                )
            assert sorted(result) == sorted(expected_files)

    def test_exclude_content_wildcard(self, setup_logger, mock_os_walk_txt):
        """Given wildcard content exclude pattern, files with matching content should be excluded."""
        # Given
        base_dir = "/path/to"
        extensions = [".txt"]
        includes = []
        excludes = []
        exclude_contents = ["*exclude*"]
        
        with patch('jet.utils.file.content_matches_pattern') as mock_content:
            # Mock to return True (match) only for files with "exclude" in name
            mock_content.side_effect = lambda path, pats: any(
                pat.endswith("*exclude*") and "test" in os.path.basename(path) 
                for pat in pats
            )
            expected_files = [
                "/path/to/docs/file1.txt", 
                "/path/to/docs/other.txt",
                "/path/to/docs/subfolder/file2.txt",
                "/path/to/scripts/script.txt"
            ]  # Exclude test_file.txt
            
            with patch("os.walk", mock_os_walk_txt):
                result = search_files(
                    base_dir, extensions, 
                    include_files=includes, 
                    exclude_files=excludes,
                    exclude_contents=exclude_contents
                )
            assert sorted(result) == sorted(expected_files)

    def test_combined_content_filters(self, setup_logger):
        """Given combined content include/exclude patterns, should apply both filters."""
        # Given
        base_dir = "/path/to"
        extensions = [".txt"]
        includes = []
        excludes = []
        include_contents = ["required"]
        exclude_contents = ["forbidden"]
        
        # Custom walk that includes specific test files
        def custom_walk(dir_path):
            if dir_path.endswith("path/to"):
                return [
                    ("/path/to/docs", [], [
                        "required_file.txt", 
                        "forbidden_file.txt", 
                        "neutral_file.txt"
                    ])
                ]
            return []
        
        with patch('jet.utils.file.content_matches_pattern') as mock_content:
            def content_side_effect(path, patterns):
                basename = os.path.basename(path)
                # For include_contents: match if pattern "required" and file contains "required_file"
                if "required" in patterns:
                    return "required_file" in basename
                # For exclude_contents: match if pattern "forbidden" and file contains "forbidden_file"  
                if "forbidden" in patterns:
                    return "forbidden_file" in basename
                return False
            mock_content.side_effect = content_side_effect
            
            expected_files = ["/path/to/docs/required_file.txt"]
            
            with patch("os.walk", custom_walk):
                result = search_files(
                    base_dir, extensions, 
                    include_files=includes, 
                    exclude_files=excludes,
                    include_contents=include_contents, 
                    exclude_contents=exclude_contents
                )
            assert sorted(result) == sorted(expected_files)

    def test_content_filter_order(self, setup_logger, mock_os_walk_txt):
        """Content filters should be applied after path filters, preserving order."""
        # Given
        base_dir = "/path/to"
        extensions = [".txt"]
        includes = ["docs"]  # Path filter first - matches docs and docs/subfolder
        excludes = []
        include_contents = ["valid"]
        
        with patch('jet.utils.file.content_matches_pattern') as mock_content:
            mock_content.side_effect = lambda path, pats: True  # All pass content check
            
            with patch("os.walk", mock_os_walk_txt):
                result = search_files(
                    base_dir, extensions, 
                    include_files=includes,
                    include_contents=include_contents
                )
                # Should include ALL docs files (including subfolders) since path filter matches "docs" anywhere
                expected_docs_files = [
                    "/path/to/docs/file1.txt", 
                    "/path/to/docs/other.txt", 
                    "/path/to/docs/subfolder/file2.txt"
                ]
                assert sorted(result) == sorted(expected_docs_files)

    def test_content_filter_with_path_filter_precedence(self, setup_logger, mock_os_walk_txt):
        """Path filters should take precedence over content filters (applied first)."""
        # Given
        base_dir = "/path/to"
        extensions = [".txt"]
        includes = ["scripts"]  # Only scripts directory - more specific
        excludes = []
        include_contents = ["*"]  # Would include everything if applied first
        
        with patch('jet.utils.file.content_matches_pattern') as mock_content:
            mock_content.side_effect = lambda path, pats: True  # All pass content check
            
            with patch("os.walk", mock_os_walk_txt):
                result = search_files(
                    base_dir, extensions, 
                    include_files=includes,
                    include_contents=include_contents
                )
                # Should only return scripts files despite content filter allowing all
                expected_files = ["/path/to/scripts/script.txt"]
                assert sorted(result) == sorted(expected_files)

    def test_empty_content_patterns(self, setup_logger, mock_os_walk_txt):
        """Empty content patterns should not affect path-based filtering."""
        # Given
        base_dir = "/path/to"
        extensions = [".txt"]
        includes = ["docs"]  # Path filter matches docs and docs/subfolder
        excludes = []
        include_contents = []  # Empty
        exclude_contents = []  # Empty
        
        with patch('jet.utils.file.content_matches_pattern') as mock_content:
            # Content function shouldn't be called with empty patterns
            mock_content.side_effect = lambda path, pats: False if not pats else True
            
            with patch("os.walk", mock_os_walk_txt):
                result = search_files(
                    base_dir, extensions, 
                    include_files=includes,
                    include_contents=include_contents,
                    exclude_contents=exclude_contents
                )
                # Should match path filter only (docs and docs/subfolder)
                expected_files = [
                    "/path/to/docs/file1.txt", 
                    "/path/to/docs/other.txt", 
                    "/path/to/docs/subfolder/file2.txt"
                ]
                assert sorted(result) == sorted(expected_files)
                # Verify content_matches_pattern was NOT called for empty patterns
                assert mock_content.call_count == 0

@pytest.fixture(autouse=True)
def cleanup():
    yield
    # Add any cleanup if needed
