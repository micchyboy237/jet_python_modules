import pytest
import sys
from jet.utils.inspect_utils import get_entry_file_path

@pytest.fixture
def mock_main_module(tmp_path):
    """Fixture to create a temporary file and set it as the main module."""
    original_main = sys.modules.get("__main__")
    file_path = tmp_path / "test_script.py"
    file_path.write_text("# Test script")
    sys.modules["__main__"] = type(sys)("__main__")
    sys.modules["__main__"].__file__ = str(file_path)
    yield file_path
    if original_main is not None:
        sys.modules["__main__"] = original_main
    else:
        del sys.modules["__main__"]

class TestGetEntryFilePath:
    """Test suite for get_entry_file_path function."""
    
    def test_get_entry_file_path_valid_with_extension(self, mock_main_module, monkeypatch):
        """Given a valid main module file in an included path, when calling get_entry_file_path with remove_extension=False, then return the full absolute path."""
        # Given
        monkeypatch.setattr("jet.utils.inspect_utils.INCLUDE_PATHS", [str(mock_main_module.parent)])
        monkeypatch.setattr("jet.utils.inspect_utils.EXCLUDE_PATHS", ["site-packages"])
        
        # When
        result = get_entry_file_path(remove_extension=False)
        expected = str(mock_main_module.resolve())
        
        # Then
        assert result == expected
    
    def test_get_entry_file_path_valid_without_extension(self, mock_main_module, monkeypatch):
        """Given a valid main module file in an included path, when calling get_entry_file_path with remove_extension=True, then return the absolute path without extension."""
        # Given
        monkeypatch.setattr("jet.utils.inspect_utils.INCLUDE_PATHS", [str(mock_main_module.parent)])
        monkeypatch.setattr("jet.utils.inspect_utils.EXCLUDE_PATHS", ["site-packages"])
        
        # When
        result = get_entry_file_path(remove_extension=True)
        expected = str(mock_main_module.with_suffix(""))
        
        # Then
        assert result == expected
    
    def test_get_entry_file_path_invalid_path(self, mock_main_module, monkeypatch):
        """Given a main module file in an excluded path, when calling get_entry_file_path, then return None."""
        # Given
        monkeypatch.setattr("jet.utils.inspect_utils.INCLUDE_PATHS", ["/other/path"])
        monkeypatch.setattr("jet.utils.inspect_utils.EXCLUDE_PATHS", [str(mock_main_module.parent)])
        
        # When
        result = get_entry_file_path(remove_extension=False)
        expected = None
        
        # Then
        assert result == expected
    
    def test_get_entry_file_path_no_main_module(self, monkeypatch):
        """Given no main module file, when calling get_entry_file_path, then return None."""
        # Given
        monkeypatch.delitem(sys.modules, "__main__", raising=False)
        
        # When
        result = get_entry_file_path(remove_extension=False)
        expected = None
        
        # Then
        assert result == expected
    
    def test_get_entry_file_path_no_file_attribute(self, monkeypatch):
        """Given a main module without a file attribute, when calling get_entry_file_path, then return None."""
        # Given
        mock_main_module = type(sys)("__main__")
        monkeypatch.setitem(sys.modules, "__main__", mock_main_module)
        
        # When
        result = get_entry_file_path(remove_extension=False)
        expected = None
        
        # Then
        assert result == expected
