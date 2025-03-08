import unittest

from jet.utils.text import fix_and_unidecode


class TestFixAndUnidecode(unittest.TestCase):
    def test_normal_unicode(self):
        sample = "Chromium\\n\\u2554"
        expected = "Chromium\\n+"
        result = fix_and_unidecode(sample)
        self.assertEqual(result, expected)

    def test_normal_unicode_2(self):
        sample = "Café"
        expected = "Cafe"
        result = fix_and_unidecode(sample)
        self.assertEqual(result, expected)

    def test_multiple_escapes(self):
        sample = "Hello \\u2603 World! \\nNew Line"
        expected = "Hello  World! \\nNew Line"
        result = fix_and_unidecode(sample)
        self.assertEqual(result, expected)

    def test_no_escapes(self):
        sample = "Simple text without escapes"
        expected = "Simple text without escapes"
        result = fix_and_unidecode(sample)
        self.assertEqual(result, expected)

    def test_mixed_escaped_and_plain(self):
        sample = "Plain text \\n with \\u03A9 Omega"
        expected = "Plain text \\n with O Omega"
        result = fix_and_unidecode(sample)
        self.assertEqual(result, expected)

    def test_double_escaped(self):
        sample = "Double escape \\\\u2554"
        expected = "Double escape \\+"
        result = fix_and_unidecode(sample)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
