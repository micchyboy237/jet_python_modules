import unittest
from unittest.mock import patch
import json
import unidecode
from jet.utils.commands import copy_to_clipboard
from jet.transformers.object import make_serializable


class TestCopyToClipboard(unittest.TestCase):
    @patch("subprocess.run")
    def test_copy_string(self, mock_subprocess):
        text = "Hello, World!"
        copy_to_clipboard(text)

        _, kwargs = mock_subprocess.call_args
        self.assertEqual(kwargs["input"], text)
        self.assertEqual(kwargs["env"], {'LANG': 'en_US.UTF-8'})

    @patch("subprocess.run")
    def test_copy_unicode_string(self, mock_subprocess):
        text = "Café"
        expected = "Cafe"  # Unidecoded result
        copy_to_clipboard(text)

        _, kwargs = mock_subprocess.call_args
        self.assertEqual(kwargs["input"], expected)
        self.assertEqual(kwargs["env"], {'LANG': 'en_US.UTF-8'})

    @patch("subprocess.run")
    def test_copy_list(self, mock_subprocess):
        data = ["apple", "banana", "cherry"]
        expected = json.dumps(data, indent=2, ensure_ascii=False)

        copy_to_clipboard(data)

        _, kwargs = mock_subprocess.call_args
        self.assertEqual(kwargs["input"], expected)
        self.assertEqual(kwargs["env"], {'LANG': 'en_US.UTF-8'})


if __name__ == "__main__":
    unittest.main()
