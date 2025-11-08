from langchain_text_splitters import MarkdownTextSplitter

from jet.file.utils import save_file
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def markdown_text_splitter_example(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    Split markdown using ``MarkdownTextSplitter`` (recursive character splitter
    that prefers markdown heading boundaries).

    Args:
        text: Input markdown string.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks.
        output_dir: Directory where chunk files are saved.

    Returns:
        List of ``Document`` objects.
    """
    splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    docs: list[str] = splitter.split_text(text)
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
results = markdown_text_splitter_example(SAMPLE_MD)
save_file(results, f"{OUTPUT_DIR}/results.json")
