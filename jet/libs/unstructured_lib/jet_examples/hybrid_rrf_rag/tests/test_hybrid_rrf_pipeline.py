import tempfile
from pathlib import Path

import pytest
from jet.libs.unstructured_lib.jet_examples.hybrid_rrf_rag.hybrid_rrf_pipeline import (
    Document,
    DocumentProcessor,
    HybridRRFPipeline,
    reciprocal_rank_fusion,
)


class TestDocumentProcessor:
    def test_load_and_chunk_directory_given_sample_local_files_when_processed_then_returns_expected_chunks(
        self,
    ):
        # Given: real-world example documents (RAG + hybrid search topics)
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            (data_dir / "rag_intro.txt").write_text(
                "Retrieval-Augmented Generation combines large language models with Unstructured partitioning of local files."
            )
            (data_dir / "rrf_explain.txt").write_text(
                "Reciprocal Rank Fusion (RRF) is the standard method to combine BM25 keyword search and vector similarity for better local file retrieval."
            )

            processor = DocumentProcessor()

            # When
            elements = processor.load_directory(str(data_dir))
            chunks = processor.chunk_elements(elements)

            # Then
            result_texts = [c.page_content.strip() for c in chunks]
            expected = [
                "Retrieval-Augmented Generation combines large language models with Unstructured partitioning of local files.",
                "Reciprocal Rank Fusion (RRF) is the standard method to combine BM25 keyword search and vector similarity for better local file retrieval.",
            ]
            assert result_texts == expected


class TestRRF:
    def test_reciprocal_rank_fusion_given_two_result_lists_when_fused_then_reranks_correctly(
        self,
    ):
        # Given: sample ranked results
        doc_a = Document(page_content="RRF is great for hybrid search")
        doc_b = Document(page_content="Unstructured excels at local file parsing")
        vector_res = [doc_a, doc_b]
        bm25_res = [doc_b, doc_a]

        # When
        fused = reciprocal_rank_fusion([vector_res, bm25_res], weights=[0.7, 0.3])

        # Then
        assert fused[0].page_content == doc_a.page_content


class TestHybridRRFPipeline:
    def test_ingest_and_query_given_sample_docs_when_queried_then_relevant_doc_ranks_high(
        self,
    ):
        # Given: real-world style files
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            (data_dir / "doc1.txt").write_text(
                "Unstructured is excellent for parsing many local file types into clean chunks."
            )
            (data_dir / "doc2.txt").write_text(
                "Hybrid RRF search combines semantic embeddings with BM25 keyword matching using Reciprocal Rank Fusion."
            )

            pipeline = HybridRRFPipeline(
                input_dir=str(data_dir),
                persist_dir=str(Path(tmp) / "chroma_test"),
            )

            # When
            pipeline.ingest()
            results = pipeline.query("What is RRF in hybrid search?", k=2)

            # Then
            result_texts = [r.page_content.strip() for r in results]
            expected_snippets = [
                "Hybrid RRF search combines semantic embeddings with BM25 keyword matching using Reciprocal Rank Fusion.",
                "Unstructured is excellent for parsing many local file types into clean chunks.",
            ]
            assert len(result_texts) == 2
            for snippet in expected_snippets:
                assert any(snippet in t for t in result_texts)

    def test_query_raises_error_when_not_ingested(self):
        # Given / When / Then
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = HybridRRFPipeline(input_dir=str(Path(tmp) / "nodata"))
            with pytest.raises(ValueError, match="Call ingest() first"):
                pipeline.query("test")
