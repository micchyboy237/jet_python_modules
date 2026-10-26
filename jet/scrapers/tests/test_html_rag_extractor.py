"""
Tests for HtmlRagExtractor.
"""

from jet.scrapers.html_rag_extractor import HtmlRagChunk, HtmlRagExtractor

SAMPLE_HTML = """
<html>
<head><title>Test Page</title></head>
<body>
    <nav>Navigation Menu</nav>
    <article>
        <h1>Main Article Title</h1>
        <p>This is the first paragraph of the main content. It contains important information.</p>
        <p>This is the second paragraph. It continues the discussion on the topic.</p>
    </article>
    <footer>Copyright 2026</footer>
</body>
</html>
"""


def test_extract_and_chunk_basic():
    extractor = HtmlRagExtractor(chunk_size=100, chunk_overlap=10, min_chunk_size=5)

    chunks = extractor.extract_and_chunk(SAMPLE_HTML, url="http://example.com")

    assert len(chunks) > 0
    assert isinstance(chunks[0], HtmlRagChunk)

    # Check that navigation and footer are likely excluded or minimized
    combined_content = " ".join([c.content for c in chunks])
    assert "Main Article Title" in combined_content
    assert "first paragraph" in combined_content

    # Check metadata
    assert chunks[0].metadata["source_url"] == "http://example.com"
    assert chunks[0].metadata["extraction_method"] == "trafilatura"


def test_extract_and_chunk_empty():
    extractor = HtmlRagExtractor()
    chunks = extractor.extract_and_chunk("")
    assert len(chunks) == 0


def test_extract_metadata():
    extractor = HtmlRagExtractor()
    doc = extractor.extract_metadata(SAMPLE_HTML, url="http://example.com")

    assert doc is not None
    # Trafilatura often prefers the main content heading (h1) over the <title> tag
    # if it appears more relevant to the extracted body.
    assert doc.title == "Main Article Title"
    assert doc.url == "http://example.com"


def test_chunk_properties():
    extractor = HtmlRagExtractor(chunk_size=50, chunk_overlap=0, min_chunk_size=1)

    chunks = extractor.extract_and_chunk(SAMPLE_HTML, doc_id="test-123")

    for chunk in chunks:
        assert chunk.doc_id == "test-123"
        assert chunk.num_tokens > 0
        assert chunk.start_idx >= 0
        assert chunk.end_idx > chunk.start_idx
        assert chunk.chunk_index >= 0
