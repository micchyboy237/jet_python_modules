from pathlib import Path
from typing import Dict
import pytest
from jet.code.html_to_markdown_analyzer import (
    convert_html_to_markdown,
    analyze_markdown_file,
    process_html_for_analysis,
    MarkdownAnalysis
)


@pytest.fixture
def temp_md_file(tmp_path: Path) -> Path:
    return tmp_path / "test.md"


class TestConvertHtmlToMarkdown:
    def test_converts_simple_html_to_markdown(self, temp_md_file: Path):
        # Given
        html_input = "<h1>Title</h1><p>Content with <a href='http://test.com'>link</a>.</p>"
        expected_content = "# Title\n\nContent with [link](http://test.com).\n"

        # When
        convert_html_to_markdown(html_input, temp_md_file)
        result_content = temp_md_file.read_text(encoding='utf-8')

        # Then
        assert result_content == expected_content


class TestAnalyzeMarkdownFile:
    def test_analyzes_markdown_with_headers_and_code(self, temp_md_file: Path):
        # Given
        md_content = """
# Header 1
Paragraph text with a [Link](https://example.com).
## Header 2
```python
print("Test")
```
"""
        temp_md_file.write_text(md_content, encoding='utf-8')
        expected_result: MarkdownAnalysis = {
            'headers': 2,
            'paragraphs': 1,
            'links': 1,
            'code_blocks': 1
        }

        # When
        result = analyze_markdown_file(temp_md_file)

        # Then
        assert result == expected_result


class TestProcessHtmlForAnalysis:
    def test_processes_html_to_analysis(self, temp_md_file: Path):
        # Given
        html_input = """
<h1>Main</h1>
<p>Text with <a href='https://example.com'>link</a>.</p>
<pre><code>code here</code></pre>
"""
        expected_result: MarkdownAnalysis = {
            'headers': 1,
            'paragraphs': 1,
            'links': 0,
            'code_blocks': 1
        }

        # When
        result = process_html_for_analysis(html_input, temp_md_file)

        # Then
        assert result == expected_result
