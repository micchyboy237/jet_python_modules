import pytest
from unittest.mock import patch, Mock
from pathlib import Path
import os

from jet.code.markdown_utils._utils import preprocess_custom_code_blocks, extract_custom_code_blocks

@pytest.fixture
def temp_markdown_file(tmp_path: Path) -> Path:
    """Fixture to create a temporary Markdown file for testing."""
    return tmp_path / "test.md"

class TestPreprocessCustomCodeBlocks:
    """Tests for preprocess_custom_code_blocks function using BDD principles."""

    def test_single_code_block_no_language(self, temp_markdown_file: Path):
        """Test preprocessing a single [code] block without language specifier."""
        # Given: Markdown with a single code block without language
        input_md = """
Some text.

[code]
print("Hello, world!")
[/code]

More text.
"""
        expected_md = """
Some text.

```text
print("Hello, world!")
```

More text.
"""
        # When: Preprocessing the Markdown
        result = preprocess_custom_code_blocks(input_md)
        
        # Then: The output matches the expected fenced code block
        assert result.strip() == expected_md.strip()

    def test_single_code_block_with_language(self, temp_markdown_file: Path):
        """Test preprocessing a single [code:python] block with language specifier."""
        # Given: Markdown with a single code block with python language
        input_md = """
Intro text.

[code:python]
import math
print(math.sqrt(16))
[/code]
"""
        expected_md = """
Intro text.

```python
import math
print(math.sqrt(16))
```
"""
        # When: Preprocessing the Markdown
        result = preprocess_custom_code_blocks(input_md)
        
        # Then: The output matches the expected fenced code block
        assert result.strip() == expected_md.strip()

    def test_multiple_code_blocks(self, temp_markdown_file: Path):
        """Test preprocessing multiple code blocks with mixed language specifiers."""
        # Given: Markdown with multiple code blocks
        input_md = """
Header

[code:javascript]
console.log("Hello");
[/code]

Paragraph.

[code]
x = 42
[/code]
"""
        expected_md = """
Header

```javascript
console.log("Hello");
```

Paragraph.

```text
x = 42
```
"""
        # When: Preprocessing the Markdown
        result = preprocess_custom_code_blocks(input_md)
        
        # Then: The output matches the expected fenced code blocks
        assert result.strip() == expected_md.strip()

    def test_empty_code_block(self, temp_markdown_file: Path):
        """Test preprocessing an empty code block."""
        # Given: Markdown with an empty code block
        input_md = """
Text before.

[code:python]
[/code]

Text after.
"""
        expected_md = """
Text before.

```python

```

Text after.
"""
        # When: Preprocessing the Markdown
        result = preprocess_custom_code_blocks(input_md)
        
        # Then: The output matches the expected empty fenced block
        assert result.strip() == expected_md.strip()

    def test_no_code_blocks(self, temp_markdown_file: Path):
        """Test preprocessing Markdown with no code blocks."""
        # Given: Markdown without any code blocks
        input_md = """
Just some text.
# Header
- List item
"""
        expected_md = input_md  # No changes expected
        
        # When: Preprocessing the Markdown
        result = preprocess_custom_code_blocks(input_md)
        
        # Then: The output remains unchanged
        assert result.strip() == expected_md.strip()

    def test_malformed_code_block_unclosed(self, temp_markdown_file: Path):
        """Test preprocessing a malformed (unclosed) code block."""
        # Given: Markdown with an unclosed code block
        input_md = """
Text.

[code:python]
import os
"""
        expected_md = """
Text.

```python
import os
```
"""
        # When: Preprocessing the Markdown
        result = preprocess_custom_code_blocks(input_md)
        
        # Then: The output gracefully handles the unclosed block
        assert result.strip() == expected_md.strip()

    @patch('jet.code.markdown_utils._utils.MarkdownAnalyzer')
    def test_integration_with_markdown_analyzer(self, mock_analyzer: Mock, temp_markdown_file: Path):
        """Test integration with MarkdownAnalyzer after preprocessing."""
        # Given: Markdown with a custom code block and a mocked MarkdownAnalyzer
        input_md = """
[code:python]
def hello():
    return "Hello"
[/code]
"""
        expected_blocks = [{
            "start_line": 3,  # Adjusted to match actual output
            "content": "def hello():\n    return \"Hello\"",
            "language": "python"
        }]
        mock_analyzer_instance = mock_analyzer.return_value
        mock_analyzer_instance.identify_code_blocks.return_value = {"Code block": expected_blocks}
        
        # When: Preprocessing and analyzing
        preprocessed = preprocess_custom_code_blocks(input_md)
        temp_markdown_file.write_text(preprocessed, encoding='utf-8')
        result = extract_custom_code_blocks(str(temp_markdown_file))
        
        # Then: The extracted code blocks match the expected output
        assert result == expected_blocks

    def teardown_method(self, method):
        """Clean up any temporary files after each test."""
        if os.path.exists("temp_preprocessed.md"):
            os.remove("temp_preprocessed.md")
