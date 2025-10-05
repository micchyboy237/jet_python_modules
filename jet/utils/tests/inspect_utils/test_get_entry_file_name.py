import pytest
from unittest.mock import patch
from jet.utils.inspect_utils import get_entry_file_name

@pytest.fixture
def mock_sys_modules():
    with patch("sys.modules") as mock_modules:
        yield mock_modules

class TestGetEntryFileName:
    def test_get_entry_file_name_valid(self, mock_sys_modules):
        # Given: A valid main module with a file path
        expected = "script.py"
        mock_sys_modules["__main__"] = type('Module', (), {'__file__': '/path/to/Jet_Projects/script.py'})

        # When: Getting the entry file name
        result = get_entry_file_name()

        # Then: The correct file name is returned
        assert result == expected

    def test_get_entry_file_name_no_main_module(self, mock_sys_modules):
        # Given: No __main__ module in sys.modules
        expected = "server"
        mock_sys_modules.get.side_effect = KeyError

        # When: Getting the entry file name
        result = get_entry_file_name()

        # Then: The default value "server" is returned
        assert result == expected

    def test_get_entry_file_name_no_file_attribute(self, mock_sys_modules):
        # Given: A main module without __file__ attribute
        expected = "server"
        mock_sys_modules["__main__"] = type('Module', (), {})

        # When: Getting the entry file name
        result = get_entry_file_name()

        # Then: The default value "server" is returned
        assert result == expected