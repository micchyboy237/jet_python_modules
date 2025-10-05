import pytest
from unittest.mock import patch
from jet.utils.inspect_utils import get_entry_file_path

@pytest.fixture
def mock_sys_modules():
    with patch("sys.modules") as mock_modules:
        yield mock_modules

@pytest.fixture
def mock_validate_filepath():
    with patch("jet.utils.inspect_utils.validate_filepath") as mock_validate:
        yield mock_validate

class TestGetEntryFilePath:
    def test_get_entry_file_path_valid(self, mock_sys_modules, mock_validate_filepath):
        # Given: A valid main module with a file path in an included directory
        expected = "/path/to/Jet_Projects/script.py"
        mock_sys_modules["__main__"] = type('Module', (), {'__file__': expected})
        mock_validate_filepath.return_value = True

        # When: Getting the entry file path
        result = get_entry_file_path()

        # Then: The absolute file path is returned
        assert result == expected

    def test_get_entry_file_path_invalid_path(self, mock_sys_modules, mock_validate_filepath):
        # Given: A main module with a file path not in included directories
        expected = None
        mock_sys_modules["__main__"] = type('Module', (), {'__file__': '/path/to/other/script.py'})
        mock_validate_filepath.return_value = False

        # When: Getting the entry file path
        result = get_entry_file_path()

        # Then: None is returned
        assert result == expected

    def test_get_entry_file_path_no_main_module(self, mock_sys_modules):
        # Given: No __main__ module in sys.modules
        expected = None
        mock_sys_modules.get.side_effect = KeyError

        # When: Getting the entry file path
        result = get_entry_file_path()

        # Then: None is returned
        assert result == expected

    def test_get_entry_file_path_no_file_attribute(self, mock_sys_modules):
        # Given: A main module without __file__ attribute
        expected = None
        mock_sys_modules["__main__"] = type('Module', (), {})

        # When: Getting the entry file path
        result = get_entry_file_path()

        # Then: None is returned
        assert result == expected