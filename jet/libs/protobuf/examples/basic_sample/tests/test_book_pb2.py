"""Tests for book protobuf operations."""
import pytest
from pathlib import Path

from src.book_example import create_sample_book, load_and_validate_book
from src.serialization_utils import save_proto_to_file


class TestBookOperations:
    """Test book protobuf map and repeated field handling."""
    
    @pytest.fixture
    def sample_book(self):
        """Sample book fixture."""
        return create_sample_book(
            title="Test Book",
            metadata={"test": "value"},
            tags=["tag1", "tag2"]
        )
    
    def test_map_serialization(self, sample_book):
        # Given
        expected_keys = ["test"]
        expected_tags = ["tag1", "tag2"]
        
        # When
        temp_file = Path("test_book.bin")
        save_proto_to_file(sample_book, temp_file)
        
        # Then
        loaded = load_and_validate_book(temp_file)
        result_keys = list(loaded.metadata.keys())
        result_tags = list(loaded.tags)
        
        assert loaded.title == "Test Book"
        assert set(result_keys) == set(expected_keys)
        assert result_tags == expected_tags
        
        temp_file.unlink()  # Cleanup
    
    def test_reserved_fields_ignored(self, sample_book):
        # Given: Try to access reserved field (should fail or be ignored)
        # When & Then
        with pytest.raises(AttributeError):
            _ = sample_book.old_field  # Reserved field access fails
    
    def test_empty_metadata(self):
        # Given
        empty_book = create_sample_book("Empty", {}, [])
        
        # When & Then
        assert len(empty_book.metadata) == 0
        assert len(empty_book.tags) == 0
        assert empty_book.title == "Empty"