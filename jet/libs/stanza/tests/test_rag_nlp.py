from jet.libs.stanza.rag_nlp import RAGPipeline, split_by_markdown_headers, chunk_markdown_sections, is_valid_sentence


EXAMPLE_MARKDOWN = """
# Introduction
This is an example markdown document. It shows how sections are split.

## Details
The system uses sliding window chunking. It also applies MMR retrieval.

### More Info
You can use this for web-scraped documents converted to markdown.
"""

class TestMarkdownSplit:
    def test_split_headers(self):
        sections = split_by_markdown_headers(EXAMPLE_MARKDOWN)
        assert len(sections) == 3
        assert sections[0]["header"] == "Introduction"
        assert sections[1]["level"] == 2

class TestChunking:
    def test_chunk_markdown_sections(self):
        chunks = chunk_markdown_sections(EXAMPLE_MARKDOWN, max_tokens=20)
        assert len(chunks) >= 3
        assert all(c.text for c in chunks)
        assert all(c.section_title for c in chunks)

class TestSentenceValidation:
    def test_is_valid_sentence(self):
        assert is_valid_sentence("This is a valid sentence.") is True
        assert is_valid_sentence("Hi.") is False

class TestPipelineIntegration:
    def setup_method(self):
        self.pipeline = RAGPipeline()

    def test_prepare_chunks_and_retrieve(self):
        chunks = self.pipeline.prepare_chunks(EXAMPLE_MARKDOWN)
        assert len(chunks) > 0
        results = self.pipeline.retrieve("How does it chunk?", chunks, top_k=2)
        assert len(results) <= 2
        assert all(hasattr(r, "text") for r in results)
