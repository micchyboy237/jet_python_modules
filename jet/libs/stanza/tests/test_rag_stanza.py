import pytest
from jet.libs.stanza import rag_stanza

# ---------- FIXTURES ----------

@pytest.fixture(scope="session")
def nlp():
    """Fixture: load lightweight English Stanza pipeline once for all tests."""
    return rag_stanza.build_stanza_pipeline(lang="en", use_gpu=False)

@pytest.fixture
def sample_text():
    return "John Doe founded OpenAI in San Francisco. The company focuses on artificial intelligence research."

@pytest.fixture
def parsed_sentences(nlp, sample_text):
    return rag_stanza.parse_sentences(sample_text, nlp)


# ---------- TEST CLASS: parse_sentences ----------

class TestParseSentences:
    def test_returns_sentence_structure(self, parsed_sentences):
        # Given: sample text with two sentences
        # When: parsed with Stanza pipeline
        result = parsed_sentences

        # Then: should return list of structured dicts with required keys
        assert isinstance(result, list)
        assert all(isinstance(s, dict) for s in result)
        for s in result:
            assert "text" in s
            assert "tokens" in s
            assert "entities" in s
            assert isinstance(s["tokens"], list)
            assert isinstance(s["entities"], list)

    def test_entities_have_offsets_and_lemmas(self, parsed_sentences):
        # Given: parsed sentences
        # When: inspecting first entity
        entities = [e for s in parsed_sentences for e in s["entities"]]
        result = entities[0] if entities else None

        # Then: entity contains text, type, start/end, lemma
        if result:
            expected_keys = {"text", "type", "start_char", "end_char", "lemma"}
            assert expected_keys.issubset(result.keys())


# ---------- TEST CLASS: build_context_chunks ----------

class TestBuildContextChunks:
    def test_builds_chunks_with_salience_and_entities(self, parsed_sentences):
        # Given: parsed sentences
        # When: building chunks
        result = rag_stanza.build_context_chunks(parsed_sentences, max_tokens=50)

        # Then: returns list of dicts with expected structure
        assert isinstance(result, list)
        assert all("salience" in c for c in result)
        assert all("entities" in c for c in result)
        assert all(isinstance(c["salience"], (int, float)) for c in result)

    def test_chunk_length_respects_max_tokens(self, parsed_sentences):
        # Given: parsed sentences and small max token limit
        # When: building chunks
        result = rag_stanza.build_context_chunks(parsed_sentences, max_tokens=10)

        # Then: no chunk exceeds limit by more than tolerance
        tolerance = 10
        assert all(c["tokens"] <= 10 + tolerance for c in result)


# ---------- TEST CLASS: _finalize_chunk_v2 ----------

class TestFinalizeChunkV2:
    def test_salience_and_offsets_computed(self, parsed_sentences):
        # Given: small list of parsed sentences
        sample = parsed_sentences[:1]

        # When: finalizing chunk
        result = rag_stanza._finalize_chunk_v2(sample, start_idx=0)

        # Then: should compute salience and offsets
        assert "salience" in result
        assert isinstance(result["salience"], (float, int))
        assert "start_char" in result
        assert "end_char" in result


# ---------- TEST CLASS: rerank_chunks_for_query ----------

class TestRerankChunksForQuery:
    def test_reranking_increases_relevance_for_matching_query(self, nlp, parsed_sentences):
        # Given: build chunks and a matching query
        chunks = rag_stanza.build_context_chunks(parsed_sentences, max_tokens=50)
        query = "Who founded OpenAI?"

        # When: reranking chunks
        result = rag_stanza.rerank_chunks_for_query(chunks, query, nlp, top_k=5)

        # Then: top chunk should contain entity 'OpenAI'
        assert isinstance(result, list)
        assert "final_score" in result[0]
        top_chunk = result[0]
        assert any("OpenAI" in e for e in top_chunk["entities"])

    def test_reranking_returns_sorted_results(self, nlp, parsed_sentences):
        # Given: query unrelated to text
        chunks = rag_stanza.build_context_chunks(parsed_sentences, max_tokens=50)
        query = "Basketball scores"

        # When: reranking
        result = rag_stanza.rerank_chunks_for_query(chunks, query, nlp, top_k=3)

        # Then: results sorted by final_score descending
        scores = [c["final_score"] for c in result]
        expected = sorted(scores, reverse=True)
        assert scores == expected
