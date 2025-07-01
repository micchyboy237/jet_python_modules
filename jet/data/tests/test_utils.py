import json
import uuid
import pytest
import hashlib
import time
from jet.data.utils import generate_unique_id, generate_key, generate_unique_hash, hash_text


class TestGenerateUniqueId:
    def test_generate_unique_id_format(self) -> None:
        """Test that generate_unique_id produces a correctly formatted UUID v4 string."""
        # Given
        expected_length = 36  # Standard UUID v4 length (8-4-4-4-12)

        # When
        result = generate_unique_id()

        # Then
        assert len(result) == expected_length
        assert uuid.UUID(result, version=4)  # Validates UUID v4 format

    def test_generate_unique_id_uniqueness(self) -> None:
        """Test that multiple calls to generate_unique_id produce unique UUIDs."""
        # Given
        num_ids = 10
        expected_unique_ids = set()

        # When
        result_ids = [generate_unique_id() for _ in range(num_ids)]

        # Then
        assert len(set(result_ids)) == num_ids
        for id_ in result_ids:
            assert len(id_) == 36  # Ensure each ID is a valid UUID length


class TestGenerateKey:
    def test_generate_key_with_args(self):
        """Test generate_key with positional arguments."""
        expected = 36  # Length of UUID string
        result = len(generate_key("test", 123))
        assert result == expected, f"Expected key length {expected}, got {result}"

        # Verify sorting behavior
        key1 = generate_key("test", 123)
        time.sleep(0.01)  # Ensure timestamp difference
        key2 = generate_key("test", 123)
        expected_order = [key1, key2]
        result_order = sorted([key2, key1])
        assert result_order == expected_order, f"Expected {expected_order}, got {result_order}"

    def test_generate_key_with_kwargs(self):
        """Test generate_key with keyword arguments."""
        expected = 36  # Length of UUID string
        result = len(generate_key(name="test", value=456))
        assert result == expected, f"Expected key length {expected}, got {result}"

        # Verify sorting behavior
        key1 = generate_key(name="test", value=456)
        time.sleep(0.01)
        key2 = generate_key(name="test", value=456)
        expected_order = [key1, key2]
        result_order = sorted([key2, key1])
        assert result_order == expected_order, f"Expected {expected_order}, got {result_order}"

    def test_generate_key_invalid_input(self):
        """Test generate_key with non-serializable input."""
        with pytest.raises(ValueError) as exc_info:
            generate_key(lambda x: x)  # Non-serializable input
        expected = "Invalid argument provided"
        result = str(exc_info.value)
        assert expected in result, f"Expected error message containing {expected}, got {result}"


class TestGenerateUniqueHash:
    def test_generate_unique_hash(self):
        """Test generate_unique_hash produces a valid UUID."""
        expected = 36  # Length of UUID v4 string
        result = len(generate_unique_hash())
        assert result == expected, f"Expected hash length {expected}, got {result}"


class TestHashText:
    def test_hash_text_string(self):
        """Test hash_text with a single string."""
        input_text = "hello"
        expected = hashlib.sha256(json.dumps(input_text).encode()).hexdigest()
        result = hash_text(input_text)
        assert result == expected, f"Expected hash {expected}, got {result}"

    def test_hash_text_list(self):
        """Test hash_text with a list of strings."""
        input_text = ["hello", "world"]
        expected = hashlib.sha256(json.dumps(
            input_text, sort_keys=True).encode()).hexdigest()
        result = hash_text(input_text)
        assert result == expected, f"Expected hash {expected}, got {result}"
