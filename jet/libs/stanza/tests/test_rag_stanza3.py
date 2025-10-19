import pytest
from jet.libs.stanza.rag_stanza3 import build_stanza_pipeline, parse_sentences, sentence_salience_score, chunk_sentences_for_rag, build_context_chunks, transformer_token_counter

# NOTE: these tests assume stanza English models are installed.
SAMPLE_TEXT = (
    "OpenAI announced a new model today. The model outperforms baseline systems on several benchmarks. "
    "Researchers from Stanford and MIT collaborated on the evaluation. "
    "It was trained on web-scale data and fine-tuned for instruction following."
)

class TestRagStanzaPipeline:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Given: a pipeline ready for English
        self.pipeline = build_stanza_pipeline(lang="en")
        yield
        # Cleanup: no persistent resources in this case

    def test_parse_sentences_outputs_expected_fields(self):
        # Given: sample text and pipeline
        sents = parse_sentences(SAMPLE_TEXT, self.pipeline)

        # When: we inspect the first sentence
        first = sents[0]

        # Then: result contains expected keys and types
        result_keys = set(first.keys())
        expected_keys = {"text", "tokens", "lemmas", "pos", "deps", "constituency", "entities"}
        assert result_keys == expected_keys, f"Expected keys {expected_keys}, got {result_keys}"

        # Flexible shape checks
        result = {"num_sentences": len(sents), "first_tokens_len": len(first["tokens"])}
        expected_num_sentences = 4
        assert result["num_sentences"] == expected_num_sentences, f"Expected {expected_num_sentences} sentences, got {result['num_sentences']}"
        assert 4 <= result["first_tokens_len"] <= 8, f"First sentence token count {result['first_tokens_len']} outside expected range [4, 8]"

    def test_sentence_salience_scores_order(self):
        # Given: parse results
        sents = parse_sentences(SAMPLE_TEXT, self.pipeline)

        # When: compute salience for each
        scores = [sentence_salience_score(s) for s in sents]

        # Then: expect scores to be reasonable and a list of floats
        result = {"scores": scores}
        expected = {"num_scores": 4, "all_floats": all(isinstance(x, float) for x in scores)}
        assert len(result["scores"]) == expected["num_scores"], f"Expected {expected['num_scores']} scores, got {len(result['scores'])}"
        assert expected["all_floats"], "Not all salience scores are floats"
        # Check: sentence with most entities has a high salience score
        entity_counts = [len(s["entities"]) for s in sents]
        max_entity_idx = max(range(len(sents)), key=lambda i: entity_counts[i])
        median_score = sorted(scores)[len(scores) // 2]
        result = {"max_entity_score": scores[max_entity_idx]}
        expected = {"min_score": median_score}
        assert result["max_entity_score"] >= expected["min_score"], (
            f"Sentence with most entities (index {max_entity_idx}) has score {result['max_entity_score']:.2f}, "
            f"below median score {expected['min_score']:.2f}"
        )
        # Check: highest score corresponds to a sentence with entities or high content
        max_score_idx = max(range(len(sents)), key=lambda i: scores[i])
        content_pos = {"NOUN", "PROPN", "VERB", "ADJ"}
        content_count = sum(1 for p in sents[max_score_idx]["pos"] if p in content_pos)
        assert len(sents[max_score_idx]["entities"]) > 0 or content_count >= 4, (
            f"Highest salience sentence (index {max_score_idx}, score {scores[max_score_idx]:.2f}) "
            "should have entities or at least 4 content-rich tokens"
        )

    def test_sentence_salience_length_contribution(self):
        # Given: parse results
        sents = parse_sentences(SAMPLE_TEXT, self.pipeline)

        # When: compute salience for each and find longest sentence
        scores = [sentence_salience_score(s) for s in sents]
        longest_sent_idx = max(range(len(sents)), key=lambda i: len(sents[i]["tokens"]))

        # Then: verify length contributes to score
        result = {"longest_score": scores[longest_sent_idx]}
        # Calculate score without length factor for comparison
        s = sents[longest_sent_idx]
        num_entities = len(s["entities"])
        content_pos = {"NOUN", "PROPN", "VERB", "ADJ"}
        content_count = sum(1 for p in s["pos"] if p in content_pos)
        root_indicator = 1.0 if any(d["deprel"].lower() == "root" for d in s["deps"]) else 0.0
        score_without_length = num_entities * 2.0 + (content_count / 10.0) + root_indicator
        expected = {"min_score_increase": score_without_length + 0.1}  # Length adds at least 0.1 (e.g., 5 tokens)
        assert result["longest_score"] > score_without_length, (
            f"Longest sentence score {result['longest_score']:.2f} should exceed score without length "
            f"{score_without_length:.2f}"
        )
        assert result["longest_score"] >= expected["min_score_increase"], (
            f"Longest sentence score {result['longest_score']:.2f} should be at least "
            f"{expected['min_score_increase']:.2f}"
        )

    def test_chunking_behaviour_basic(self):
        # Given: parsed sentences and transformer token counter
        sents = parse_sentences(SAMPLE_TEXT, self.pipeline)
        token_counter = transformer_token_counter(model_name="bert-base-uncased")

        # When: chunk with small max_tokens to force multiple chunks
        chunks = chunk_sentences_for_rag(sents, max_tokens=6, overlap=2, token_counter=token_counter)

        # Then: expect multiple chunks and valid metadata
        result = {"num_chunks": len(chunks), "first_chunk_sent_indices": chunks[0]["sentence_indices"]}
        expected = {"min_chunks": 2}
        assert result["num_chunks"] >= expected["min_chunks"], f"Expected at least {expected['min_chunks']} chunks, got {result['num_chunks']}"
        assert all(isinstance(i, int) for i in result["first_chunk_sent_indices"]), "First chunk sentence indices must be integers"
        assert "salience" in chunks[0]["metadata"], "First chunk missing salience metadata"
        # Additional check: verify transformer token counting
        assert chunks[0]["est_token_count"] == token_counter(sents[chunks[0]["sentence_indices"][0]]["tokens"]), "Token count mismatch in first chunk"

def test_build_context_chunks_top_level():
    # Given: sample text and transformer token counter
    token_counter = transformer_token_counter(model_name="bert-base-uncased")

    # When: run high-level chunking
    chunks = build_context_chunks(SAMPLE_TEXT, lang="en", max_tokens=20, overlap=5, token_counter=token_counter)

    # Then: expect at least 2 chunks
    result = {"chunks": len(chunks)}
    expected = {"min_chunks": 2}
    assert result["chunks"] >= expected["min_chunks"], f"Expected at least {expected['min_chunks']} chunks, got {result['chunks']}"
    # Additional check: verify token counts
    assert all(chunk["est_token_count"] <= 20 for chunk in chunks), "Some chunks exceed max_tokens"
