import unittest
from jet.code.markdown_code_extractor import MarkdownCodeExtractor, CodeBlock


class TestMarkdownCodeExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = MarkdownCodeExtractor()

    def test_extract_single_code_block_with_file_path(self):
        message = """#### File Path: `example.py`
```python
def hello_world():
    print("Hello, World!")
```"""
        result = self.extractor.extract_code_blocks(message)
        expected = [
            CodeBlock(
                code='def hello_world():\n    print("Hello, World!")',
                language="python",
                file_path="example.py",
            )
        ]
        self.assertEqual(result, expected)

    def test_extract_multiple_code_blocks_with_and_without_file_paths(self):
        message = """#### File Path: `script1.py`
```python
print("Script 1")
```
```bash
echo "Script 2"
```"""
        result = self.extractor.extract_code_blocks(message)
        expected = [
            CodeBlock(
                code='print("Script 1")',
                language="python",
                file_path="script1.py",
            ),
            CodeBlock(
                code='echo "Script 2"',
                language="bash",
                file_path=None,
            ),
        ]
        self.assertEqual(result, expected)

    def test_no_code_blocks(self):
        message = """This is just text with no code blocks."""
        result = self.extractor.extract_code_blocks(message)
        self.assertEqual(result, [])

    def test_code_block_with_unknown_language(self):
        message = """#### File Path: `unknown_script`
```
print("Unknown language")
```"""
        result = self.extractor.extract_code_blocks(message)
        expected = [
            CodeBlock(
                code='print("Unknown language")',
                language="python",
                file_path="unknown_script",
            )
        ]
        self.assertEqual(result, expected)

    def test_extract_code_block_with_file_path_pattern(self):
        message = """
    File Path: /path/to/script.py
    ```python
    def hello_world():
        print("Hello, world!")
    """
        expected = [
            CodeBlock(
                code='    def hello_world():\n        print("Hello, world!")',
                language='python',
                file_path='/path/to/script.py'
            )
        ]
        result = self.extractor.extract_code_blocks(message)
        self.assertEqual(result, expected)

    def test_no_file_path_prefix(self):
        message = """
    test.py
    ```python
    def hello_world():
        print("Hello, world!")
    ```
    """
        expected = [
            CodeBlock(
                code='    def hello_world():\n        print("Hello, world!")',
                language="python",
                file_path="test.py"
            )
        ]
        result = self.extractor.extract_code_blocks(message)
        self.assertEqual(result, expected)

    def test_no_file_path_detected(self):
        message = """

    ```python
    def hello_world():
        print("Hello, world!")
    ```
    """
        expected = [
            CodeBlock(
                code='    def hello_world():\n        print("Hello, world!")',
                language="python",
                file_path=None
            )
        ]
        result = self.extractor.extract_code_blocks(message)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
