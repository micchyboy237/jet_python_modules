import unittest

from jet.utils.object import truncate_strings


class TestTruncateStrings(unittest.TestCase):
    def test_simple_string(self):
        self.assertEqual(truncate_strings("a" * 200, 100), "a" * 100)
    
    def test_nested_dict(self):
        data = {"a": "x" * 150, "b": {"c": "y" * 120}}
        result = truncate_strings(data, 100)
        self.assertEqual(result["a"], "x" * 100)
        self.assertEqual(result["b"]["c"], "y" * 100)
    
    def test_list_inside_dict(self):
        data = {"a": ["z" * 130, {"b": "m" * 200}]}
        result = truncate_strings(data, 100)
        self.assertEqual(result["a"][0], "z" * 100)
        self.assertEqual(result["a"][1]["b"], "m" * 100)
    
    def test_tuple_and_set(self):
        data = {
            "t": ("a" * 150, "b" * 50),
            "s": {"c" * 200, "d" * 20},
        }
        result = truncate_strings(data, 100)
        self.assertEqual(result["t"][0], "a" * 100)
        self.assertEqual(result["t"][1], "b" * 50)
        self.assertIn("c" * 100, result["s"])
        self.assertIn("d" * 20, result["s"])
    
    def test_non_string_values(self):
        data = {"num": 123, "bool": True, "none": None}
        result = truncate_strings(data, 100)
        self.assertEqual(result, data)  # unchanged


class TestTruncateStringsWithSuffix(unittest.TestCase):
    def test_simple_string_with_suffix(self):
        self.assertEqual(
            truncate_strings("a" * 200, 100, "..."),
            "a" * 100 + "..."
        )

    def test_string_not_truncated(self):
        s = "short string"
        self.assertEqual(truncate_strings(s, 100, "..."), s)  # unchanged

    def test_nested_dict_with_suffix(self):
        data = {"a": "x" * 150, "b": {"c": "y" * 120}}
        result = truncate_strings(data, 100, "...")
        self.assertEqual(result["a"], "x" * 100 + "...")
        self.assertEqual(result["b"]["c"], "y" * 100 + "...")

    def test_list_inside_dict_with_suffix(self):
        data = {"a": ["z" * 130, {"b": "m" * 200}]}
        result = truncate_strings(data, 100, "...")
        self.assertEqual(result["a"][0], "z" * 100 + "...")
        self.assertEqual(result["a"][1]["b"], "m" * 100 + "...")

    def test_tuple_and_set_with_suffix(self):
        data = {
            "t": ("a" * 150, "b" * 50),
            "s": {"c" * 200, "d" * 20},
        }
        result = truncate_strings(data, 100, "...")
        self.assertEqual(result["t"][0], "a" * 100 + "...")
        self.assertEqual(result["t"][1], "b" * 50)  # not truncated
        self.assertIn("c" * 100 + "...", result["s"])
        self.assertIn("d" * 20, result["s"])

    def test_non_string_values(self):
        data = {"num": 123, "bool": True, "none": None}
        result = truncate_strings(data, 100, "...")
        self.assertEqual(result, data)  # unchanged


class TestTruncateStringsWithDynamicSuffix(unittest.TestCase):
    def _suffix_factory(self, orig: str, trunc: str) -> str:
        return f"... [TRUNCATED {{{len(orig) - len(trunc)}}}]"

    def test_simple_string_dynamic_suffix(self):
        # Given
        input_str = "a" * 250
        max_len = 100
        expected_suffix = "... [TRUNCATED {150}]"
        expected = "a" * 100 + expected_suffix

        # When
        result = truncate_strings(input_str, max_len, self._suffix_factory)

        # Then
        self.assertEqual(result, expected)

    def test_nested_structure_dynamic_suffix(self):
        # Given
        data = {
            "msg": "x" * 300,
            "details": {
                "log": "y" * 180,
                "items": ["z" * 220]
            }
        }
        max_len = 100
        suffix_factory = self._suffix_factory

        # When
        result = truncate_strings(data, max_len, suffix_factory)

        # Then
        self.assertEqual(result["msg"], "x" * 100 + "... [TRUNCATED {200}]")
        self.assertEqual(result["details"]["log"], "y" * 100 + "... [TRUNCATED {80}]")
        self.assertEqual(result["details"]["items"][0], "z" * 100 + "... [TRUNCATED {120}]")

    def test_no_truncation_dynamic_suffix(self):
        # Given
        data = {"short": "hello", "num": 42}

        # When
        result = truncate_strings(data, 100, self._suffix_factory)

        # Then
        self.assertEqual(result["short"], "hello")
        self.assertEqual(result["num"], 42)


if __name__ == "__main__":
    unittest.main()
