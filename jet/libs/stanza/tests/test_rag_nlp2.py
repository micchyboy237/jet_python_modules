# test_rag_pipeline.py
from jet.libs.stanza.rag_nlp2 import (
    preprocess_markdown_to_sentences,
    chunk_sentences,
    retrieve,
)
from sentence_transformers import SentenceTransformer

class TestRAGPipeline:
    """
    Test behavior of rag pipeline utilities:
    - Given short markdown docs with headers and inline code, sentences should
      be produced and chunked sensibly.
    - Retrieval returns similarity and (when requested) diversity metadata.
    """

    def setup_method(self):
        # Given a tiny embedding model for tests we use a small sentence-transformer model.
        # In CI this should be replaced with a mock or a deterministic model.
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.doc_md = (
            "# Title\n\n"
            "This is the first paragraph. It has two sentences.\n\n"
            "## Code\n\n"
            "```python\nprint('hello')\n```\n\n"
            "Final paragraph mentioning OpenAI and testing."
        )

    def test_preprocess_and_chunk(self):
        # Given markdown text, When preprocessed, Then sentences list should contain meaningful pieces.
        sents = preprocess_markdown_to_sentences(self.doc_md, stanza_pipe=None)
        assert any("first paragraph" in s for s in sents)
        # When chunked with small max_chars, Then multiple chunks may be produced.
        chunks = chunk_sentences(sents, max_chars=40)
        assert isinstance(chunks, list)
        assert all("text" in c for c in chunks)

    def test_embedding_and_simple_retrieval(self):
        # Given chunks and a model, When embedding and retrieving without MMR,
        sents = preprocess_markdown_to_sentences(self.doc_md, stanza_pipe=None)
        chunks = chunk_sentences(sents, max_chars=200)
        # prepare query
        query = "hello from code example"
        results = retrieve(query, chunks, embedding_model=self.model, mmr=False, mmr_top_k=3, stanza_pipe=None)
        # Then we expect a list of results with similarity floats and diversity None
        assert isinstance(results, list)
        assert all(isinstance(r["score"]["similarity"], float) for r in results)
        assert all(r["score"]["diversity"] is None for r in results)

    def test_mmr_returns_diversity_scores(self):
        sents = preprocess_markdown_to_sentences(self.doc_md, stanza_pipe=None)
        chunks = chunk_sentences(sents, max_chars=200)
        results = retrieve("testing OpenAI", chunks, embedding_model=self.model, mmr=True, mmr_top_k=2, mmr_lambda=0.6, stanza_pipe=None)
        # Then diversity fields should be floats (or 0.0 for first)
        assert len(results) <= 2
        assert all(isinstance(r["score"]["similarity"], float) for r in results)
        assert any(r["score"]["diversity"] is not None for r in results)
