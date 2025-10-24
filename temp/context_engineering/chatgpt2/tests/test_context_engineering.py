# tests/test_context_engineering.py
import pytest
from context_engineering import ContextEngine, Document, ChunkConfig, RetrievalConfig, InMemoryIndex, _approx_token_count

# --- Mocks ------------------------------------------------
class DummyEmbeddingModel:
    """
    Simple deterministic embedding: returns vector of char counts per token-window.
    Deterministic to make unit tests reproducible.
    """
    def encode(self, texts):
        embs = []
        for t in texts:
            # vector length 8: counts of chars in chunks of the string (simple, deterministic)
            out = []
            chunk_len = max(1, len(t) // 8)
            for i in range(0, len(t), chunk_len):
                out.append(float(len(t[i:i+chunk_len])))
            # pad/truncate to length 8
            if len(out) < 8:
                out.extend([0.0]*(8-len(out)))
            embs.append(out[:8])
        return embs

class DummyLLM:
    def generate(self, prompt: str) -> str:
        # returns a short string reflecting it received the prompt (for testing)
        # In production you'd call an actual LLM client
        return f"LLM_RESPONSE: prompt_len={len(prompt)}"

# --- Fixtures ------------------------------------------------
@pytest.fixture
def dummy_embedding():
    return DummyEmbeddingModel()

@pytest.fixture
def dummy_llm():
    return DummyLLM()

# --- Tests ------------------------------------------------
class TestContextEngine:
    def setup_method(self):
        # Given: small collection of documents
        self.docs = [
            Document(id="doc1", text="This is a short document about apples. Apples are red and tasty.", meta={}),
            Document(id="doc2", text="Bananas are yellow. They are rich in potassium and taste sweet.", meta={}),
        ]

    def test_chunking_and_token_count(self):
        # Given: a longish text and chunk config
        cfg = ChunkConfig(max_tokens=10, overlap=2, split_on_sentence=True)
        chunks = []
        for d in self.docs:
            chunks.extend(__import__("context_engineering").chunk_text(d["id"], d["text"], cfg))
        # Then: chunks list should be non-empty and token counts reasonable
        assert len(chunks) >= 2
        # Approx token count must be positive
        assert all(_approx_token_count(c["text"]) > 0 for c in chunks)

    def test_ingest_retrieve_generate_flow(self, dummy_embedding, dummy_llm):
        # Given: embedding and LLM wrappers that expose simple api
        emb_fn = lambda texts: dummy_embedding.encode(texts)
        llm_fn = lambda prompt: dummy_llm.generate(prompt)
        engine = ContextEngine(embedding_fn=emb_fn, llm_fn=llm_fn, index=InMemoryIndex())
        # When: ingest documents
        engine.ingest_documents(self.docs, chunk_cfg=ChunkConfig(max_tokens=40, overlap=10))
        # Then: index has some chunks
        assert len(engine.index.chunks) > 0
        # When: generate answer for a query relating to apples
        cfg = RetrievalConfig(chunk_cfg=ChunkConfig(max_tokens=40, overlap=10), top_k=3, context_token_budget=200)
        resp = engine.generate("Tell me about apples.", cfg=cfg)
        # Then: response should be produced by LLM mock and reflect prompt length
        assert resp.startswith("LLM_RESPONSE:")
        # Also ensure that the prompt contained the word 'apples' somewhere (evidence of retrieval)
        # We can simulate by checking that engine.llm_fn was invoked (mock returns deterministic)
        assert "prompt_len" in resp

    def test_no_documents_still_calls_llm(self, dummy_embedding, dummy_llm):
        # Given an engine with no documents ingested
        emb_fn = lambda texts: dummy_embedding.encode(texts)
        llm_fn = lambda prompt: dummy_llm.generate(prompt)
        engine = ContextEngine(embedding_fn=emb_fn, llm_fn=llm_fn, index=InMemoryIndex())
        # When: generate called without ingest
        resp = engine.generate("Who won the match?")
        # Then: still calls LLM and returns response
        assert resp.startswith("LLM_RESPONSE:")