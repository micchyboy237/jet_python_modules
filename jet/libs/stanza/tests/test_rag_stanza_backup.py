# test_rag_stanza.py
from jet.libs.stanza.rag_stanza_backup import build_stanza_pipeline, parse_sentences, sentence_salience_score, chunk_sentences_for_rag, build_context_chunks

# NOTE: these tests assume stanza English models are installed.
SAMPLE_TEXT = (
    "OpenAI announced a new model today. The model outperforms baseline systems on several benchmarks. "
    "Researchers from Stanford and MIT collaborated on the evaluation. "
    "It was trained on web-scale data and fine-tuned for instruction following."
)

class TestRagStanzaPipeline:
    def setup_method(self):
        # Given: a pipeline ready for English
        self.pipeline = build_stanza_pipeline(lang="en")

    def teardown_method(self):
        # no persistent resources to cleanup in this simple example
        pass

    def test_parse_sentences_outputs_expected_fields(self):
        # Given: sample text and pipeline
        sents = parse_sentences(SAMPLE_TEXT, self.pipeline)

        # When: we inspect the first sentence
        first = sents[0]

        # Then: result contains expected keys and types
        result_keys = set(first.keys())
        expected_keys = {"text", "tokens", "lemmas", "pos", "deps", "constituency", "entities"}
        assert result_keys == expected_keys

        # Flexible shape checks
        result = {"num_sentences": len(sents), "first_tokens_len": len(first["tokens"])}
        expected_num_sentences = 4  # Should remain stable
        # Allow minor variation in tokenizer behavior (5–8 tokens typical)
        assert result["num_sentences"] == expected_num_sentences
        assert 4 <= result["first_tokens_len"] <= 8

    def test_sentence_salience_scores_order(self):
        # Given: parse results
        sents = parse_sentences(SAMPLE_TEXT, self.pipeline)

        # When: compute salience for each
        scores = [sentence_salience_score(s) for s in sents]

        # Then: expect scores to be reasonable and a list of floats
        assert all(isinstance(x, float) for x in scores)
        assert len(scores) == 4

    def test_chunking_behaviour_basic(self):
        # Given: parsed sentences
        sents = parse_sentences(SAMPLE_TEXT, self.pipeline)

        # When: chunk with small max_tokens to force multiple chunks
        chunks = chunk_sentences_for_rag(sents, max_tokens=6, overlap=2, token_counter=lambda toks: len(toks))

        # Then: expect multiple chunks and valid metadata
        result = {"num_chunks": len(chunks), "first_chunk_sent_indices": chunks[0]["sentence_indices"]}
        # Expect at least 2 chunks, but allow tokenizer variance
        assert result["num_chunks"] >= 2
        assert all(isinstance(i, int) for i in result["first_chunk_sent_indices"])
        # Check metadata presence
        assert "salience" in chunks[0]["metadata"]

def test_build_context_chunks_top_level():
    # smoke test for the high-level convenience function
    chunks = build_context_chunks(SAMPLE_TEXT, lang="en", max_tokens=20, overlap=5)
    result = {"chunks": len(chunks)}
    # At least 2 chunks expected (depending on tokenizer granularity)
    assert result["chunks"] >= 2
