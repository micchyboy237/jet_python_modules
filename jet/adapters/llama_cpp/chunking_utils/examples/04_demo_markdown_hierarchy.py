"""
Demo: Markdown Hierarchy Chunking

Shows how to chunk markdown documents while preserving header structure,
parent-child relationships, and precise source indices.
"""

import json
import logging

from jet.adapters.llama_cpp.chunking_utils import chunk_markdown_hierarchy_with_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    markdown_doc = """# Project Documentation

Welcome to the project overview. This guide covers setup and usage.

## Installation

First, install the dependencies using pip.
Make sure you have Python 3.9+ installed.

Then clone the repository from GitHub.

### System Requirements

You need at least 8GB RAM and 4 CPU cores.
GPU acceleration is recommended for large models.

## Configuration

Edit the config.yaml file to set your preferences.
API keys should be stored in environment variables.

# Advanced Topics

This section covers performance tuning and debugging.
"""

    logger.info("Starting markdown hierarchy chunking demo...")

    results = chunk_markdown_hierarchy_with_data(
        markdown_text=markdown_doc,
        chunk_size=64,
        chunk_overlap=8,
        min_chunk_size=10,
        show_progress=True,
    )

    logger.info(f"Generated {len(results)} hierarchical chunks:")
    for i, chunk in enumerate(results):
        print(f"\n{'=' * 60}")
        print(f"Chunk {i + 1}: [{chunk['header']}] (Level {chunk['level']})")
        print(f"Parent: {chunk['parent_header'] or '(Root)'}")
        print(f"Tokens: {chunk['num_tokens']} | Section: {chunk['section_index']}")
        print(
            f"Body Range: [{chunk['metadata']['body_start_idx']}:{chunk['metadata']['body_end_idx']}]"
        )
        print(f"Content Preview: {chunk['content'][:100]}...")

        # Verify index correctness
        reconstructed = markdown_doc[
            chunk["metadata"]["start_idx"] : chunk["metadata"]["end_idx"]
        ]
        if chunk["header"] and chunk["header"] not in reconstructed:
            logger.warning(f"Header mismatch in chunk {i + 1}!")

    # Show JSON output for first chunk
    if results:
        print(f"\n{'=' * 60}")
        print("Sample Chunk JSON:")
        print(json.dumps(results[0], indent=2))


if __name__ == "__main__":
    main()
