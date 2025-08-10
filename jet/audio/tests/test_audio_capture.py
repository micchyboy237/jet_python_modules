import pytest
import os
import glob
from unittest.mock import patch
from jet.audio.utils import get_next_file_suffix


@pytest.fixture
def temp_dir(tmp_path):
    """Fixture to create a temporary directory."""
    with patch("os.getcwd", return_value=str(tmp_path)):
        yield tmp_path


class TestGetNextFileSuffix:
    def test_no_files_exist(self, temp_dir):
        """Given no files exist, when getting next suffix, then return 0."""
        # Given
        file_prefix = "recording"
        expected = 0

        # When
        result = get_next_file_suffix(file_prefix)

        # Then
        assert result == expected, f"Expected suffix {expected}, got {result}"

    def test_existing_files_with_valid_suffixes(self, temp_dir):
        """Given files with valid suffixes, when getting next suffix, then return max + 1."""
        # Given
        file_prefix = "recording"
        files = [
            f"{file_prefix}_00000.wav",
            f"{file_prefix}_00001.wav",
            f"{file_prefix}_00003.wav"
        ]
        for f in files:
            (temp_dir / f).touch()
        expected = 4

        # When
        result = get_next_file_suffix(file_prefix)

        # Then
        assert result == expected, f"Expected suffix {expected}, got {result}"

    def test_files_with_invalid_suffixes(self, temp_dir):
        """Given files with invalid suffixes, when getting next suffix, then ignore invalid and return max + 1."""
        # Given
        file_prefix = "recording"
        files = [
            f"{file_prefix}_00000.wav",
            f"{file_prefix}_abc.wav",
            f"{file_prefix}_.wav",
            f"{file_prefix}_00002.wav"
        ]
        for f in files:
            (temp_dir / f).touch()
        expected = 3

        # When
        result = get_next_file_suffix(file_prefix)

        # Then
        assert result == expected, f"Expected suffix {expected}, got {result}"

    def test_empty_suffix(self, temp_dir):
        """Given a file with empty suffix after underscore, when getting next suffix, then ignore it."""
        # Given
        file_prefix = "recording"
        files = [f"{file_prefix}_.wav"]
        for f in files:
            (temp_dir / f).touch()
        expected = 0

        # When
        result = get_next_file_suffix(file_prefix)

        # Then
        assert result == expected, f"Expected suffix {expected}, got {result}"
