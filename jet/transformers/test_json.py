import unittest
from jet.transformers.json_parsers import parse_json


class TestParseJson(unittest.TestCase):

    def test_valid_json(self):
        sample = '{"key": "value"}'
        expected = {"key": "value"}
        result = parse_json(sample)
        self.assertEqual(result, expected)

    def test_invalid_json(self):
        sample = '{key: value}'  # Invalid JSON (missing quotes around keys)
        expected = sample  # Should return the original string
        result = parse_json(sample)
        self.assertEqual(result, expected)

    def test_valid_dict(self):
        sample = {"key": "value"}
        expected = {"key": "value"}
        result = parse_json(sample)
        self.assertEqual(result, expected)

    def test_valid_list(self):
        sample = [{"key": "value"}]
        expected = [{"key": "value"}]
        result = parse_json(sample)
        self.assertEqual(result, expected)

    def test_empty_string(self):
        sample = ''
        expected = ''  # Should return the original empty string
        result = parse_json(sample)
        self.assertEqual(result, expected)

    def test_non_string_input(self):
        sample = 12345  # Integer input
        with self.assertRaises(TypeError):
            parse_json(sample)

    def test_json_array(self):
        sample = '[1, 2, 3]'
        expected = [1, 2, 3]
        result = parse_json(sample)
        self.assertEqual(result, expected)

    def test_json_nested_object(self):
        sample = '{"a": {"b": {"c": 1}}}'
        expected = {"a": {"b": {"c": 1}}}
        result = parse_json(sample)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
