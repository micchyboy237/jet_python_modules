"""
01_demo_basic_parse.py
Demonstrates basic document parsing, element inspection, and RAG context extraction.
"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is in path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from jet.adapters.unstructured.document_parser import parse_document


def create_sample_markdown() -> str:
    """Create a temporary sample markdown file for demonstration."""
    content = """# Sample Document Title

## Introduction
This is a sample narrative text used to demonstrate the unstructured adapter.
It contains multiple sentences. Some are short. Others are significantly longer 
to test how the parser handles varying lengths of text within a single paragraph.

## Key Features
- Automatic structure detection
- Token-aware chunking
- Metadata preservation

## Code Example
```python
def hello():
    print("Hello from parsed code snippet")
```

## Conclusion
Parsing complete. This section verifies that headers reset chunk boundaries 
in structured documents.
"""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


def main():
    print("=" * 60)
    print("DEMO 01: Basic Document Parsing & RAG Context Extraction")
    print("=" * 60)

    # Create a sample file since we don't want to depend on external assets
    sample_path = create_sample_markdown()
    print(f"\n📄 Created sample file: {sample_path}")

    try:
        # Parse with explicit chunk size for predictable demo output
        result = parse_document(
            path=sample_path,
            chunk_max_tokens=150,
            chunk_overlap_tokens=20,
            model=None,  # Use heuristic token counting for demo portability
        )

        # Display summary statistics
        print("\n" + "-" * 40)
        print("PARSE SUMMARY")
        print("-" * 40)
        print(f"Status:         {result['status']}")
        print(f"Element Count:  {result['element_count']}")
        print(f"Categories:     {result['categories']}")
        print(f"Word Count:     {result['word_count']}")
        print(f"Chunk Count:    {len(result['chunks'])}")

        # Show first few elements
        print("\n" + "-" * 40)
        print("FIRST 3 ELEMENTS")
        print("-" * 40)
        for elem in result["elements"][:3]:
            print(f"  [{elem.get('type', 'Unknown')}] {elem.get('text', '')[:80]}...")

        # Show RAG context preview
        rag_ctx = result["rag_context"]
        print("\n" + "-" * 40)
        print(f"RAG CONTEXT PREVIEW ({len(rag_ctx)} chars)")
        print("-" * 40)
        print(rag_ctx[:500])
        if len(rag_ctx) > 500:
            print("... [truncated]")

        # Show chunk details
        print("\n" + "-" * 40)
        print("CHUNK DETAILS")
        print("-" * 40)
        for i, chunk in enumerate(result["chunks"]):
            print(f"\n  Chunk {i + 1}:")
            print(f"    Strategy:    {chunk['strategy']}")
            print(f"    Tokens:      {chunk['token_count']}")
            print(f"    Text Preview: {chunk['text'][:100].replace(chr(10), ' ')}...")
            meta = chunk.get("metadata", {})
            if meta.get("element_ids"):
                print(
                    f"    Element IDs: {meta['element_ids'][:3]}{'...' if len(meta['element_ids']) > 3 else ''}"
                )

    finally:
        # Cleanup temp file
        if os.path.exists(sample_path):
            os.unlink(sample_path)
            print(f"\n🗑️  Cleaned up temp file: {sample_path}")

    print("\n✅ Demo 01 Complete")


if __name__ == "__main__":
    main()
