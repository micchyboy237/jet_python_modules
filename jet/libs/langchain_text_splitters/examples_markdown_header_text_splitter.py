from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from jet.file.utils import save_file
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def markdown_header_text_splitter_example(
    text: str,
    headers_to_split_on: list[tuple[str, str]] | None = None,
    return_each_line: bool = False,
    strip_headers: bool = True,
) -> list[Document]:
    """
    Split markdown by explicit header levels, attaching header values as metadata.

    Args:
        text: Input markdown string.
        headers_to_split_on: ``[(prefix, metadata_key), ...]``; ``None`` uses defaults.
        return_each_line: Return a ``Document`` per line (with shared metadata).
        strip_headers: Remove the header line from chunk content.
        output_dir: Directory where chunk files are saved.

    Returns:
        List of ``Document`` objects.
    """
    if headers_to_split_on is None:
        headers_to_split_on = [
            ("# ", "Header 1"),
            ("## ", "Header 2"),
            ("### ", "Header 3"),
        ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        return_each_line=return_each_line,
        strip_headers=strip_headers,
    )
    docs: list[Document] = splitter.split_text(text)
    return docs


# Example

SAMPLE_MD = """# Project Overview

This is the top-level introduction.

## Installation

```bash
pip install my-lib
```

## Usage

### Basic Example

```python
from my_lib import hello
print(hello())
```

### Advanced Example

See the docs.

## API Reference

Detailed function signatures.
"""

print("Running MarkdownTextSplitter example...")
results = markdown_header_text_splitter_example(SAMPLE_MD)
save_file(results, f"{OUTPUT_DIR}/results.json")
