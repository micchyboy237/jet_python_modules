import pytest
import shutil
from unittest.mock import patch

from jet.llm.logger_utils import ChatLogger, get_next_file_counter


@pytest.fixture
def temp_log_dir(tmp_path):
    """Create a temporary log directory for testing."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir  # Return Path object instead of string


@pytest.fixture
def mock_event_settings():
    """Mock EventSettings to return a fixed start_time."""
    with patch("shared.setup.events.EventSettings.get_entry_event") as mock:
        mock.return_value = {"start_time": "2025-09-08"}
        yield mock


class TestGetNextFileCounter:
    """Tests for get_next_file_counter function."""

    def test_empty_directory(self, temp_log_dir):
        """Given an empty log directory, when getting the next file counter, then it returns 1."""
        # Given
        method = "stream_chat"
        expected = 1

        # When
        result = get_next_file_counter(str(temp_log_dir), method)

        # Then
        assert result == expected, f"Expected counter {expected}, but got {result}"

    def test_existing_files_single_method(self, temp_log_dir):
        """Given files with the same method, when getting the next file counter, then it returns the highest prefix + 1."""
        # Given
        method = "stream_chat"
        files = ["01_stream_chat.json", "03_stream_chat.json", "02_stream_chat.json"]
        for file in files:
            (temp_log_dir / file).touch()
        expected = 4

        # When
        result = get_next_file_counter(str(temp_log_dir), method)

        # Then
        assert result == expected, f"Expected counter {expected}, but got {result}"

    def test_mixed_method_files(self, temp_log_dir):
        """Given files with different methods, when getting the next file counter, then it ignores other methods."""
        # Given
        method = "stream_chat"
        files = ["01_stream_chat.json", "02_chat.json", "03_generate.json"]
        for file in files:
            (temp_log_dir / file).touch()
        expected = 2

        # When
        result = get_next_file_counter(str(temp_log_dir), method)

        # Then
        assert result == expected, f"Expected counter {expected}, but got {result}"

    def test_invalid_filenames(self, temp_log_dir):
        """Given files with invalid prefixes, when getting the next file counter, then it ignores them."""
        # Given
        method = "stream_chat"
        files = ["01_stream_chat.json",
                 "invalid_stream_chat.json", "ab_stream_chat.json"]
        for file in files:
            (temp_log_dir / file).touch()
        expected = 2

        # When
        result = get_next_file_counter(str(temp_log_dir), method)

        # Then
        assert result == expected, f"Expected counter {expected}, but got {result}"


class TestChatLoggerInitialization:
    """Tests for ChatLogger initialization with file counter."""

    def test_new_directory_initial_counter(self, temp_log_dir, mock_event_settings):
        """Given a new log directory, when initializing ChatLogger, then the counter is 1."""
        # Given
        method = "stream_chat"
        expected = 1

        # When
        logger = ChatLogger(log_dir=str(temp_log_dir), method=method)
        result = logger._file_counter

        # Then
        assert result == expected, f"Expected counter {expected}, but got {result}"

    def test_existing_files_counter(self, temp_log_dir, mock_event_settings):
        """Given existing files for the method, when initializing ChatLogger, then the counter is set to highest prefix + 1."""
        # Given
        method = "stream_chat"
        files = ["01_stream_chat.json", "03_stream_chat.json", "02_stream_chat.json"]
        sub_dir = temp_log_dir / "2025-09-08"
        sub_dir.mkdir()
        for file in files:
            (sub_dir / file).touch()
        expected = 4

        # When
        logger = ChatLogger(log_dir=str(temp_log_dir), method=method)
        result = logger._file_counter

        # Then
        assert result == expected, f"Expected counter {expected}, but got {result}"

    def test_mixed_method_files_counter(self, temp_log_dir, mock_event_settings):
        """Given mixed method files, when initializing ChatLogger, then the counter only considers the relevant method."""
        # Given
        method = "stream_chat"
        files = ["01_stream_chat.json", "02_chat.json", "03_generate.json"]
        sub_dir = temp_log_dir / "2025-09-08"
        sub_dir.mkdir()
        for file in files:
            (sub_dir / file).touch()
        expected = 2

        # When
        logger = ChatLogger(log_dir=str(temp_log_dir), method=method)
        result = logger._file_counter

        # Then
        assert result == expected, f"Expected counter {expected}, but got {result}"


@pytest.fixture(autouse=True)
def cleanup_temp_files(tmp_path):
    """Clean up temporary files and directories after each test."""
    yield
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
