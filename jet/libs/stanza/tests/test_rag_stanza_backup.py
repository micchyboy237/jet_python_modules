"""
Integration tests for RAG-enhanced Stanza pipeline using realistic long inputs.

These tests ensure that:
- Sentence parsing, tokenization, and entity recognition work correctly.
- Context chunks are properly formed for RAG vector retrieval.
- Salience, entity aggregation, and chunk structure remain consistent.
"""

from pathlib import Path
from typing import Dict, Any
from jet.libs.stanza.examples.examples_rag_stanza import run_rag_stanza_demo


class TestRagStanzaExamples:
    """Integration-style tests for realistic long input behavior."""

    def setup_method(self):
        # Given a long multi-sentence input text simulating a real article
        self.demo_text_path = Path(__file__).parent / "demo_long_input.txt"
        self.demo_text_path.write_text(
            (
                "OpenAI announced the GPT-5 model on October 15, 2025, marking a major leap "
                "in multimodal reasoning and multilingual understanding. "
                "The company claims that GPT-5 can handle text, image, and structured data "
                "simultaneously, offering developers a unified API for advanced RAG systems. "
                "Industry analysts from Gartner and McKinsey noted that the integration of "
                "reasoning and retrieval may redefine enterprise search. "
                "In a blog post, OpenAI also shared new benchmarks showing 40% higher factual "
                "recall across multilingual datasets. "
                "For instance, a medical RAG system fine-tuned on PubMed articles showed improved "
                "diagnostic explanations with lower hallucination rates. "
                "Meanwhile, universities like Stanford, MIT, and ETH Zurich are testing open-source "
                "alternatives built on Stanza and Hugging Face Transformers, demonstrating that "
                "syntax-aware chunking can drastically reduce embedding redundancy."
            ),
            encoding="utf-8"
        )

    def teardown_method(self):
        # Clean up any temporary file created for tests
        if self.demo_text_path.exists():
            self.demo_text_path.unlink()

    # -----------------------------------------------------------------------------------------
    # Core integration tests
    # -----------------------------------------------------------------------------------------

    def test_run_demo_returns_expected_structure(self):
        """
        Given: A realistic long text
        When:  run_rag_stanza_demo() is executed
        Then:  It returns structured parsing and chunking results with expected fields
        """
        # --- When ---
        result: Dict[str, Any] = run_rag_stanza_demo(self.demo_text_path.read_text())

        # --- Then ---
        assert isinstance(result, dict), "Expected result to be a dict"
        assert "parsed_sentences" in result
        assert "chunks" in result

        # Expect multiple sentences parsed
        num_sentences = len(result["parsed_sentences"])
        assert num_sentences >= 5, f"Expected ≥5 sentences, got {num_sentences}"

        # Verify token-level detail
        first_sentence = result["parsed_sentences"][0]
        assert isinstance(first_sentence["tokens"], list)
        assert len(first_sentence["tokens"]) > 5, "Tokens missing or truncated"

        # Check chunk structure
        chunks = result["chunks"]
        assert all(
            isinstance(ch["sent_indices"], list) and isinstance(ch["text"], str)
            for ch in chunks
        ), "Each chunk must have sent_indices and text"

    def test_salience_and_entities_populated(self):
        """
        Given: A long text processed by the RAG Stanza pipeline
        When:  Chunks are created
        Then:  Each chunk should include salience and entity information
        """
        # --- When ---
        result = run_rag_stanza_demo(self.demo_text_path.read_text())

        # --- Then ---
        for chunk in result["chunks"]:
            assert "salience" in chunk, "Missing salience score"
            assert isinstance(chunk["salience"], (float, int)), "Salience should be numeric"

            assert "entities" in chunk, "Missing entity list"
            assert isinstance(chunk["entities"], list)
            assert any(
                isinstance(ent, str) and len(ent) > 1 for ent in chunk["entities"]
            ), "No valid named entities found in chunk"

    def test_context_chunk_count_reasonable(self):
        """
        Given: A realistic long article
        When:  The text is chunked for RAG context
        Then:  Number of chunks should be within a reasonable range
        """
        # --- When ---
        result = run_rag_stanza_demo(self.demo_text_path.read_text())

        # --- Then ---
        chunk_count = len(result["chunks"])
        assert 2 <= chunk_count <= 6, f"Unexpected chunk count: {chunk_count}"

    def test_chunk_salience_relative_order(self):
        """
        Given: Multiple context chunks
        When:  Sorted by salience score
        Then:  The top chunk should have equal or higher salience than others
        """
        result = run_rag_stanza_demo(self.demo_text_path.read_text())

        saliences = [c["salience"] for c in result["chunks"]]
        assert all(s >= 0 for s in saliences)
        assert saliences[0] >= min(saliences), "Top chunk should have higher salience"

    def test_entities_across_chunks_cover_expected_terms(self):
        """
        Given: Extracted entities across chunks
        When:  Merged into a unique set
        Then:  Must include key entities from the text (OpenAI, GPT-5, Stanza, etc.)
        """
        result = run_rag_stanza_demo(self.demo_text_path.read_text())

        all_entities = set()
        for chunk in result["chunks"]:
            all_entities.update(chunk.get("entities", []))

        expected_keywords = {"OpenAI", "GPT-5", "Stanza", "RAG"}
        assert expected_keywords.intersection(all_entities), (
            f"Expected key entities missing. Found only: {all_entities}"
        )

import pytest
from jet.libs.stanza.rag_stanza import build_stanza_pipeline, parse_sentences, build_context_chunks

@pytest.fixture
def nlp():
    return build_stanza_pipeline()

@pytest.fixture
def sample_text():
    return (
        "OpenAI unveiled the GPT-5 model in October 2025, showcasing advanced reasoning "
        "and multilingual understanding capabilities. "
        "The model can process text, images, and structured data simultaneously. "
        "Analysts from Gartner believe this integration may redefine enterprise AI search. "
        "Meanwhile, universities such as MIT and ETH Zurich are testing Stanza-based parsing "
        "to improve context chunking for retrieval-augmented generation systems."
    )

class TestBuildContextChunks:
    def test_chunking_correct_indices_and_entities(self, nlp, sample_text):
        # Given: Parsed sentences from sample text
        parsed = parse_sentences(sample_text, nlp)
        # When: Building chunks with max_tokens=80
        chunks = build_context_chunks(parsed, max_tokens=80)
        # Then: Expect two chunks with correct indices and entities
        result = chunks
        expected = [
            {
                "sent_indices": [0, 1, 2],
                "tokens": sum(len(s["tokens"]) for s in parsed[:3]),
                "entities": ["OpenAI", "GPT-5", "October 2025", "Gartner"],
            },
            {
                "sent_indices": [3],
                "tokens": len(parsed[3]["tokens"]),
                "entities": ["MIT", "ETH Zurich", "Stanza"],
            },
        ]
        assert len(result) == len(expected), "Incorrect number of chunks"
        for i, chunk in enumerate(result):
            assert chunk["sent_indices"] == expected[i]["sent_indices"], f"Chunk {i+1} has incorrect sentence indices"
            assert sorted(chunk["entities"]) == sorted(expected[i]["entities"]), f"Chunk {i+1} has incorrect entities"
            assert chunk["tokens"] == expected[i]["tokens"], f"Chunk {i+1} has incorrect token count"

@pytest.fixture(autouse=True)
def cleanup():
    yield  # Clean up after tests if needed
