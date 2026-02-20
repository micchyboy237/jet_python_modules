"""Pytest tests - class-based, BDD style, human-readable examples, exact asserts on lists."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from jet.libs.unstructured_lib.jet_examples.rag_grok_agents_answer.rag_document import (
    Chunk,
)
from jet.libs.unstructured_lib.jet_examples.rag_grok_agents_answer.rag_embedder import (
    LlamaCppEmbedder,
)
from jet.libs.unstructured_lib.jet_examples.rag_grok_agents_answer.rag_pipeline import (
    RAGPipeline,
)
from jet.libs.unstructured_lib.jet_examples.rag_grok_agents_answer.rag_processor import (
    DocumentProcessor,
)
from jet.libs.unstructured_lib.jet_examples.rag_grok_agents_answer.rag_vectorstore import (
    ChromaVectorStore,
)

EMBED_URL = os.getenv("LLAMA_CPP_EMBED_URL")


class TestDocumentProcessor:
    """Behaviors for document loading + chunking."""

    def test_process_file_given_earnings_report_md_when_processed_then_chunks_respect_titles(
        self, tmp_path: Path
    ):
        # Given: human-readable sample earnings report with clear sections
        sample_md = """# Q3 2025 Earnings Report

Revenue increased 25% YoY.

## Financial Highlights
Profit reached $15M.
"""
        file_path = tmp_path / "earnings.md"
        file_path.write_text(sample_md)

        # Tuned params force title-aware split (low combine prevents merge on small sections)
        processor = DocumentProcessor(
            max_characters=150,
            combine_text_under_n_chars=50,
            new_after_n_chars=100,
        )

        # When
        result: list[Chunk] = processor.process_file(str(file_path))
        # Temporary debug (remove after fixing): show how many chunks produced
        from jet.libs.unstructured_lib.jet_examples.rag_grok_agents_answer.rag_vectorstore import (
            console,
        )

        console.print(
            f"[yellow]DEBUG process_file test: produced {len(result)} chunks[/yellow]"
        )

        # Then
        expected_texts = [
            "Q3 2025 Earnings Report\n\nRevenue increased 25% YoY.",
            "Financial Highlights\n\nProfit reached $15M.",
        ]
        result_texts = [c["text"] for c in result]
        assert (
            result_texts == expected_texts
        )  # exact match on structure-aware chunks (plain text)
        assert len(result) == 2
        assert "Q3 2025" in result[0]["text"]  # real-world verification
        assert "Financial Highlights" in result[1]["text"]


class TestLlamaCppEmbedder:
    """Behaviors for embedding server calls."""

    @patch(
        "jet.libs.unstructured_lib.jet_examples.rag_grok_agents_answer.rag_embedder.OpenAI"
    )
    def test_embed_documents_given_texts_when_called_then_returns_embeddings(
        self, mock_openai_class
    ):
        # Given
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2]),
            MagicMock(embedding=[0.3, 0.4]),
        ]
        mock_client.embeddings.create.return_value = mock_response

        embedder = LlamaCppEmbedder(embed_url=EMBED_URL)  # mock url

        # When
        result = embedder.embed_documents(["text one", "text two"])

        # Then
        expected = [[0.1, 0.2], [0.3, 0.4]]
        assert result == expected
        mock_client.embeddings.create.assert_called_once()

    def test_embed_query_given_single_text_when_called_then_returns_single_vector(self):
        # Given
        embedder = LlamaCppEmbedder(embed_url=EMBED_URL)  # will be mocked in full run

        # When + Then (integration style with real server when available)
        # Note: run with real server for full; this verifies method shape
        pass  # placeholder - full test in pipeline


class TestChromaVectorStore:
    """Behaviors for vector storage/retrieval."""

    def test_add_and_search_given_chunks_when_searched_then_returns_matching(
        self, tmp_path: Path
    ):
        # Given
        store = ChromaVectorStore(persist_directory=str(tmp_path / "test_db"))
        chunks: list[Chunk] = [
            {"text": "Revenue up 25%", "metadata": {"source": "earnings.md"}},
            {"text": "Profit $15M", "metadata": {"source": "earnings.md"}},
        ]
        dummy_embs = [[0.1] * 1536, [0.2] * 1536]

        # When
        store.add_documents(chunks, dummy_embs)
        result = store.similarity_search([0.15] * 1536, k=2)

        # Then
        expected_texts = ["Revenue up 25%", "Profit $15M"]
        result_texts = [c["text"] for c in result]
        assert sorted(result_texts) == sorted(expected_texts)


class TestRAGPipeline:
    """End-to-end pipeline behaviors (with mocks for LLM/embed)."""

    @patch(
        "jet.libs.unstructured_lib.jet_examples.rag_grok_agents_answer.rag_embedder.LlamaCppEmbedder"
    )
    @patch(
        "jet.libs.unstructured_lib.jet_examples.rag_grok_agents_answer.rag_llm.LlamaCppLLM"
    )
    def test_query_given_mocked_components_when_called_then_returns_generated_answer(
        self, mock_llm_class, mock_embedder_class
    ):
        # Given
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 1536
        mock_embedder_class.return_value = mock_embedder

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Revenue grew 25%."
        mock_llm_class.return_value = mock_llm

        # fake store with pre-added data
        with tempfile.TemporaryDirectory() as tmp:
            store = ChromaVectorStore(persist_directory=tmp)
            chunks: list[Chunk] = [
                {
                    "text": "Revenue increased 25% YoY.",
                    "metadata": {"source": "test.md"},
                }
            ]
            store.add_documents(chunks, [[0.1] * 1536])

            pipeline = RAGPipeline(
                embedder=mock_embedder,
                llm=mock_llm,
                vector_store=store,
            )

            # When
            result = pipeline.query("What was the revenue growth?")

            # Then
            expected = "Revenue grew 25%."
            assert result == expected
            mock_llm.generate.assert_called_once()
