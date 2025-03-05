import unittest
import re
from utils import generate_key


class TestGenerateKey(unittest.TestCase):
    def setUp(self):
        self.uuid_pattern = re.compile(
            r'^[a-f0-9]{8}-[a-f0-9]{4}-[5][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$', re.IGNORECASE
        )

    def test_uuid_format(self):
        """Ensure generate_key produces a valid UUID format."""
        key = generate_key("test", 123)
        self.assertTrue(self.uuid_pattern.match(
            key), f"Invalid UUID format: {key}")

    def test_deterministic_output(self):
        """Ensure the same input always produces the same UUID."""
        key1 = generate_key("user", 12345)
        key2 = generate_key("user", 12345)
        self.assertEqual(
            key1, key2, "UUID should be deterministic for the same inputs")

    def test_different_inputs_produce_different_keys(self):
        """Ensure different inputs produce different UUIDs."""
        key1 = generate_key("user", 12345)
        key2 = generate_key("admin", 67890)
        self.assertNotEqual(
            key1, key2, "Different inputs should produce different UUIDs")

    def test_edge_case_empty_input(self):
        """Ensure function handles empty input correctly."""
        key = generate_key()
        self.assertTrue(self.uuid_pattern.match(
            key), f"Invalid UUID format for empty input: {key}")

    def test_uuid_length(self):
        """Ensure the generated UUID has the correct length of 36 characters."""
        key = generate_key("test", 456)
        self.assertEqual(len(key), 36, f"UUID length mismatch: {key}")

    def test_uuid_character_counts(self):
        """Ensure the correct number of characters appear between hyphens."""
        key = generate_key("test", 789)
        parts = key.split("-")
        expected_lengths = [8, 4, 4, 4, 12]
        actual_lengths = [len(part) for part in parts]
        self.assertEqual(actual_lengths, expected_lengths,
                         f"UUID parts length mismatch: {key}")


if __name__ == "__main__":
    unittest.main()
