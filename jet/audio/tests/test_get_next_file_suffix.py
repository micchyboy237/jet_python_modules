import pytest
import os
from unittest.mock import patch
from jet.audio.utils import get_next_file_suffix
from jet.logger import logger


class TestGetNextFileSuffix:
    test_cases = [
        (
            "no_existing_files",
            {
                "glob_return": [],
                "expected": 0
            }
        ),
        (
            "existing_files_with_suffixes",
            {
                "glob_return": [
                    "/path/to/recording_00001.wav",
                    "/path/to/recording_00002.wav"
                ],
                "expected": 3
            }
        ),
        (
            "non_matching_files",
            {
                "glob_return": [
                    "/path/to/recording_invalid.wav",
                    "/path/to/other_00001.wav"
                ],
                "expected": 0
            }
        ),
        (
            "files_with_zero_suffix",
            {
                "glob_return": ["/path/to/recording_00000.wav"],
                "expected": 1
            }
        )
    ]

    def setup_method(self):
        logger.handlers = []  # Clear handlers to avoid duplicate logs

    @pytest.mark.parametrize("test_name,case", test_cases)
    def test_get_next_file_suffix(self, test_name, case):
        """Test getting the next file suffix for various file patterns."""
        # Given: A mocked glob.glob with specific file patterns
        with patch("glob.glob") as mock_glob, patch("os.path.basename") as mock_basename:
            mock_glob.return_value = case["glob_return"]
            mock_basename.side_effect = lambda x: x.split('/')[-1]
            # When: The function is called with a file prefix
            result = get_next_file_suffix("recording")
            # Then: The result matches the expected suffix
            expected = case["expected"]
            assert result == expected, f"Expected suffix {expected}, got {result}"
