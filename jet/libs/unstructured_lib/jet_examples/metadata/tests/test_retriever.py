"""pytest tests following your exact style (BDD, result/expected vars, full list asserts, tmp_path cleanup)."""

from pathlib import Path

import pytest
from jet.libs.unstructured_lib.jet_examples.metadata.retriever import (
    RetrieverConfig,
    UnstructuredLocalRetriever,
)


@pytest.fixture
def temp_retriever_dir(tmp_path: Path):
    """Clean temporary dir + DB for each test."""
    return tmp_path


class TestUnstructuredLocalRetriever:
    """Separate test class for behaviors."""

    def test_ingest_text_file_and_retrieve(self, temp_retriever_dir: Path):
        """BDD: Given a real-world quarterly sales report text file.
        When ingested.
        Then retrieve returns exact expected content with metadata.
        """
        # Given: real-world sample (like blog PDF excerpt but .txt for easy testing)
        sample_content = (
            "Q3 2025 Sales Report\n"
            "Revenue: $2.3M (up 20%).\n"
            "Key metric: tables show regional breakdown."
        )
        sample_file = temp_retriever_dir / "q3_sales_report.txt"
        sample_file.write_text(sample_content)

        config = RetrieverConfig(
            collection_name="test_sales",
            persist_directory=str(temp_retriever_dir / "chroma"),
        )
        retriever = UnstructuredLocalRetriever(config)

        # When
        retriever.ingest_file(sample_file)

        # Then
        result = retriever.retrieve("sales revenue", top_k=1)
        # For plain .txt files, Unstructured often returns UncategorizedText
        # We care more about content retrieval + source tracking than exact category
        expected_source_file = "q3_sales_report.txt"

        assert result["documents"] == [["Revenue: $2.3M (up 20%)."]]
        assert result["metadatas"][0][0]["source_file"] == expected_source_file

    def test_ingest_directory_and_filtered_retrieve(self, temp_retriever_dir: Path):
        """BDD: Given a directory with multiple real-world doc files.
        When ingested via directory method.
        Then filtered query returns only matching elements.
        """
        # Given: two sample files (realistic content)
        (temp_retriever_dir / "tech_article.txt").write_text(
            "Advanced RAG uses element partitioning and metadata filtering."
        )
        (temp_retriever_dir / "finance_table.txt").write_text(
            "Table: Q3 Revenue by region."
        )

        config = RetrieverConfig(
            collection_name="test_dir",
            persist_directory=str(temp_retriever_dir / "chroma_dir"),
        )
        retriever = UnstructuredLocalRetriever(config)

        # When
        retriever.ingest_directory(temp_retriever_dir, pattern="**/*.txt")

        # Then: filter demo (blog-style element_type filter)
        result = retriever.retrieve(
            "revenue", top_k=2, filters={"element_type": "NarrativeText"}
        )
        result_texts = result["documents"][0]
        expected_texts = [
            "Advanced RAG uses element partitioning and metadata filtering."
        ]  # only the narrative one after filter

        assert result_texts == expected_texts  # exact list assert
        assert result["metadatas"][0][0]["source_file"] == "tech_article.txt"
