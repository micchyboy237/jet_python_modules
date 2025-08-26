import pytest
import os
import fnmatch
from typing import Literal, Optional
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


@pytest.fixture(autouse=True)
def cleanup():
    yield
    # Add any cleanup if needed
