# jet_python_modules/jet/adapters/llama_cpp/chunking_utils/tests/test_markdown.py
import pytest
from jet.adapters.llama_cpp.chunking_utils.markdown import (
    chunk_markdown_hierarchy,
    chunk_markdown_hierarchy_with_data,
)


@pytest.fixture
def sample_markdown():
    return """# Introduction
Welcome to the guide.

## Setup
Install dependencies first.
Then configure your environment.

### Advanced Config
Edit the YAML file carefully.

## Usage
Run the main script with python main.py.
"""


def test_empty_input():
    assert chunk_markdown_hierarchy("") == []
    assert chunk_markdown_hierarchy_with_data("") == []


def test_basic_hierarchy_preservation(sample_markdown):
    chunks = chunk_markdown_hierarchy_with_data(sample_markdown, chunk_size=128)
    headers = [c["header"] for c in chunks]
    assert "# Introduction" in headers
    assert "## Setup" in headers
    assert "### Advanced Config" in headers


def test_parent_child_relationships(sample_markdown):
    chunks = chunk_markdown_hierarchy_with_data(sample_markdown, chunk_size=128)
    setup_chunk = next(c for c in chunks if c["header"] == "## Setup")
    advanced_chunk = next(c for c in chunks if c["header"] == "### Advanced Config")

    assert setup_chunk["parent_header"] == "# Introduction"
    assert advanced_chunk["parent_header"] == "## Setup"
    assert advanced_chunk["parent_id"] == setup_chunk["header_doc_id"]


def test_source_indices_are_valid(sample_markdown):
    chunks = chunk_markdown_hierarchy_with_data(sample_markdown, chunk_size=64)
    for chunk in chunks:
        meta = chunk["metadata"]
        full_slice = sample_markdown[meta["start_idx"] : meta["end_idx"]]
        body_slice = sample_markdown[meta["body_start_idx"] : meta["body_end_idx"]]

        if chunk["header"]:
            assert chunk["header"] in full_slice
        assert chunk["content"] in body_slice or chunk["content"] in full_slice


def test_simple_string_output(sample_markdown):
    chunks = chunk_markdown_hierarchy(sample_markdown, chunk_size=128)
    assert all(isinstance(c, str) for c in chunks)
    assert any("# Introduction" in c for c in chunks)


def test_multiple_documents():
    docs = ["# Doc A\nContent A.", "# Doc B\nContent B."]
    results = chunk_markdown_hierarchy_with_data(docs, chunk_size=64, ids=["a", "b"])

    assert len(results) == 2
    assert results[0]["doc_id"] == "a"
    assert results[1]["doc_id"] == "b"
    assert results[0]["doc_index"] == 0
    assert results[1]["doc_index"] == 1


def test_chunk_splitting_respects_budget():
    long_body = " ".join([f"Sentence number {i}." for i in range(50)])
    text = f"# Long Section\n{long_body}"

    chunks = chunk_markdown_hierarchy_with_data(text, chunk_size=32, min_chunk_size=5)
    assert len(chunks) > 1

    # All chunks should share same header_doc_id
    header_ids = {c["header_doc_id"] for c in chunks}
    assert len(header_ids) == 1


def test_header_without_body_skipped():
    text = """# Empty Header
## Child With Body
Actual content here.
"""
    chunks = chunk_markdown_hierarchy_with_data(text, chunk_size=64)
    assert len(chunks) == 1
    assert chunks[0]["header"] == "## Child With Body"
    assert chunks[0]["parent_header"] == "# Empty Header"


def test_preamble_content_before_first_header():
    text = "Some intro text.\n# First Header\nBody."
    chunks = chunk_markdown_hierarchy_with_data(text, chunk_size=64)

    preamble = chunks[0]
    assert preamble["header"] == ""
    assert preamble["section_index"] == -1
    assert "intro text" in preamble["content"]


def test_metadata_types_correct():
    text = "# Test\nContent."
    chunks = chunk_markdown_hierarchy_with_data(text, chunk_size=64)

    chunk = chunks[0]
    assert isinstance(chunk["metadata"], dict)
    assert "start_idx" in chunk["metadata"]
    assert "body_start_idx" in chunk["metadata"]
    assert isinstance(chunk["level"], int)
    assert isinstance(chunk["num_tokens"], int)
