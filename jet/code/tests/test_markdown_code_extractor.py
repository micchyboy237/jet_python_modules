# tests/test_markdown_code_extractor.py
import unittest

from jet.code.markdown_code_extractor import (  # ← replace with actual import path
    CodeBlock,
    MarkdownCodeExtractor,
)


class TestMarkdownCodeExtractorBase(unittest.TestCase):
    """Base class with helper assertions"""

    def setUp(self):
        self.extractor = MarkdownCodeExtractor()

    def assertCodeBlockEqual(
        self,
        block: CodeBlock,
        expected_code: str,
        expected_lang: str,
        expected_path: str | None = None,
        expected_ext: str | None = None,
    ):
        self.assertEqual(block["code"], expected_code)
        self.assertEqual(block["language"], expected_lang)
        self.assertEqual(block["file_path"], expected_path)
        self.assertEqual(block["extension"], expected_ext)


class TestLanguageDetectionAndNormalization(TestMarkdownCodeExtractorBase):
    """Tests language alias handling and normalization"""

    def test_common_language_aliases(self):
        cases = [
            ("```py\nprint(1)\n```", "python", ".py"),
            ("```js\nconsole.log()\n```", "javascript", ".js"),
            ("```rb\nputs 'hi'\n```", "ruby", ".rb"),
            ("```yml\nkey: value\n```", "yaml", ".yaml"),
            ("```yaml\nkey: value\n```", "yaml", ".yaml"),
            ("```txt\nhello world\n```", "text", ".txt"),
            ("```text\nplain\n```", "text", ".txt"),
            ("```sh\necho hi\n```", "bash", ".sh"),  # if you added bash
        ]

        for markdown, expected_lang, expected_ext in cases:
            with self.subTest(markdown=markdown):
                blocks = self.extractor.extract_code_blocks(markdown)
                self.assertEqual(len(blocks), 1)
                self.assertCodeBlockEqual(
                    blocks[0],
                    code=blocks[0]["code"].strip(),
                    expected_code=blocks[0]["code"].strip(),
                    expected_lang=expected_lang,
                    expected_ext=expected_ext,
                )

    def test_no_language_defaults_to_text(self):
        md = """```
some plain content
```"""
        blocks = self.extractor.extract_code_blocks(md)
        self.assertEqual(len(blocks), 1)
        self.assertCodeBlockEqual(blocks[0], "some plain content", "text", None, ".txt")

    def test_empty_language_tag(self):
        md = """```
hello
```"""
        blocks = self.extractor.extract_code_blocks(md)
        self.assertEqual(blocks[0]["language"], "text")


class TestFilePathAssociation(TestMarkdownCodeExtractorBase):
    """Tests File Path: ... pattern detection and association"""

    def test_file_path_before_code_block(self):
        md = """File Path: `src/utils.py`
```python
def hello():
    pass
```"""
        blocks = self.extractor.extract_code_blocks(md)
        self.assertEqual(len(blocks), 1)
        self.assertCodeBlockEqual(
            blocks[0],
            "def hello():\n    pass",
            "python",
            "src/utils.py",
            ".py",
        )

    def test_file_path_with_backticks_optional(self):
        md = """File Path: src/config.yml
```yml
port: 8080
```"""
        blocks = self.extractor.extract_code_blocks(md)
        self.assertEqual(blocks[0]["file_path"], "src/config.yml")
        self.assertEqual(
            blocks[0]["extension"], ".yml"
        )  # or .yaml depending on your mapping

    def test_multiple_blocks_with_different_paths(self):
        md = """File Path: `app.py`
```py
print("app")
```

File Path: `utils/helpers.rb`
```rb
puts "helper"
```"""
        blocks = self.extractor.extract_code_blocks(md)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["file_path"], "app.py")
        self.assertEqual(blocks[1]["file_path"], "utils/helpers.rb")


class TestMultipleCodeBlocksAndText(TestMarkdownCodeExtractorBase):
    """Tests extraction of multiple blocks and text content"""

    def test_multiple_code_blocks(self):
        md = """Some intro text

```js
console.log("one")
```

Middle text

```python
print("two")
```

Final text"""
        blocks = self.extractor.extract_code_blocks(md, with_text=False)
        self.assertEqual(len(blocks), 2)

        blocks_with_text = self.extractor.extract_code_blocks(md, with_text=True)
        self.assertEqual(
            len(blocks_with_text), 5
        )  # intro + code + middle + code + final

    def test_consecutive_code_blocks(self):
        md = """```py
a = 1
```
```js
b = 2
```
```txt
c = 3
```"""
        blocks = self.extractor.extract_code_blocks(md)
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[0]["language"], "python")
        self.assertEqual(blocks[1]["language"], "javascript")
        self.assertEqual(blocks[2]["language"], "text")


class TestRemoveCodeBlocks(TestMarkdownCodeExtractorBase):
    """Tests the remove_code_blocks functionality"""

    def test_remove_code_blocks_basic(self):
        md = """Hello

```python
print("remove me")
```

World"""

        cleaned = self.extractor.remove_code_blocks(md)
        expected = """Hello

World"""
        self.assertEqual(cleaned.strip(), expected.strip())

    def test_keep_file_paths_option(self):
        md = """File Path: `script.py`
```py
print(1)
```

Some text"""

        # Without keeping paths
        cleaned_no_paths = self.extractor.remove_code_blocks(md, keep_file_paths=False)
        self.assertNotIn("File Path", cleaned_no_paths)

        # With keeping paths
        cleaned_with_paths = self.extractor.remove_code_blocks(md, keep_file_paths=True)
        self.assertIn("File Path: `script.py`", cleaned_with_paths)


class TestEdgeCases(TestMarkdownCodeExtractorBase):
    """Edge cases and robustness"""

    def test_empty_markdown(self):
        blocks = self.extractor.extract_code_blocks("")
        self.assertEqual(blocks, [])

    def test_only_text_no_code(self):
        md = "Just some\ntext content"
        blocks = self.extractor.extract_code_blocks(md, with_text=True)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["language"], "text")

    def test_code_block_with_empty_lines(self):
        md = """```python

print("hi")



x = 10
```"""
        blocks = self.extractor.extract_code_blocks(md)
        self.assertEqual(
            blocks[0]["code"].count("\n"), 5
        )  # preserves internal newlines

    def test_code_block_inside_text_with_backticks(self):
        md = """Some text with ```inline

```js
real block
```"""
        blocks = self.extractor.extract_code_blocks(md, with_text=False)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["language"], "javascript")


if __name__ == "__main__":
    unittest.main()
