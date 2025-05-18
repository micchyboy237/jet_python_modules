import unittest
import tempfile
import os
import jet.logger
from jet.code.extraction.extract_definitions import extract_class_and_function_defs

sample = '''
def greet():
    """Say hello to the user"""

class Greeter:
    """Greeter class with simple behavior"""

    def say_hi(self):
        """Print a hi message"""
'''


class TestExtractHeadersAndDocstrings(unittest.TestCase):
    def write_temp_file(self, directory: str, filename: str, content: str) -> str:
        path = os.path.join(directory, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_extracts_docstrings_only(self):
        expected = [
            "def greet():\n    \"\"\"Say hello to the user\"\"\"",
            "class Greeter:\n    \"\"\"Greeter class with simple behavior\"\"\"",
            "    def say_hi(self):\n        \"\"\"Print a hi message\"\"\""
        ]
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = self.write_temp_file(
                tmpdirname, "docstrings_only.py", sample)
            result = extract_class_and_function_defs(tmpdirname)
            flat_results = result[path]
            self.assertEqual(flat_results, expected)

    def test_sorting_by_numeric_filename(self):
        files = {
            "10_file.py": "def f10():\n    pass",
            "2_file.py": "def f2():\n    pass",
            "1_file.py": "def f1():\n    pass",
            "file_without_number.py": "def f():\n    pass",
        }
        with tempfile.TemporaryDirectory() as tmpdirname:
            for fname, content in files.items():
                self.write_temp_file(tmpdirname, fname, content)

            result = extract_class_and_function_defs(tmpdirname)

            # Extract only the basenames from keys and check the sorted order
            sorted_filenames = list(map(os.path.basename, result.keys()))

            expected_order = ["1_file.py", "2_file.py",
                              "10_file.py", "file_without_number.py"]
            self.assertEqual(sorted_filenames, expected_order)


if __name__ == "__main__":
    unittest.main()
