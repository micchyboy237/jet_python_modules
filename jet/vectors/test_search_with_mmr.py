import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from jet.vectors.search_with_mmr import (
    preprocess_texts,
    embed_search,
    rerank_results,
    mmr_diversity,
    merge_duplicate_texts_agglomerative,
    search_diverse_context,
    Header,
    PreprocessedText,
    SimilarityResult
)
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch


@pytest.fixture
def mock_sentence_transformer():
    with patch("sentence_transformers.SentenceTransformer") as MockModel:
        mock_model = MockModel.return_value
        yield mock_model


@pytest.fixture
def mock_cross_encoder():
    with patch("sentence_transformers.CrossEncoder") as MockCrossEncoder:
        mock_cross_encoder = MockCrossEncoder.return_value
        yield mock_cross_encoder


@pytest.fixture
def sample_headers():
    return [
        {
            "header": "## Introduction",
            "content": "This is the first introduction.",
            "header_level": 2,
            "parent_header": None
        },
        {
            "header": "## Intro",
            "content": "This is another intro text.",
            "header_level": 2,
            "parent_header": None
        },
        {
            "header": "## Conclusion",
            "content": "This is the conclusion.",
            "header_level": 2,
            "parent_header": None
        },
        {
            "header": "## Overview",
            "content": "This is an overview text.",
            "header_level": 2,
            "parent_header": None
        }
    ]


class TestMergeDuplicateTextsAgglomerative:
    def test_basic_deduplication(self, capsys, mock_sentence_transformer):
        texts = [
            {
                "text": "## Introduction\nThis is the first introduction.",
                "doc_index": 0,
                "id": "doc_0",
                "header_level": 2,
                "parent_header": None,
                "header": "## Introduction",
                "content": "This is the first introduction."
            },
            {
                "text": "## Intro\nThis is another intro text.",
                "doc_index": 1,
                "id": "doc_1",
                "header_level": 2,
                "parent_header": None,
                "header": "## Intro",
                "content": "This is another intro text."
            },
            {
                "text": "## Conclusion\nThis is the conclusion.",
                "doc_index": 2,
                "id": "doc_2",
                "header_level": 2,
                "parent_header": None,
                "header": "## Conclusion",
                "content": "This is the conclusion."
            }
        ]
        mock_sentence_transformer.encode.side_effect = lambda headers, **kwargs: np.array([
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0]
        ])
        result = merge_duplicate_texts_agglomerative(
            texts,
            model_name="all-MiniLM-L12-v2",
            device="mps",
            similarity_threshold=0.7,
            batch_size=32
        )
        assert len(
            result) == 2, f"Should deduplicate to 2 texts, got {len(result)}: {[t['header'] for t in result]}"
        merged_text = next(
            (t for t in result if t["header"] == "## Introduction"), None)
        assert merged_text is not None, "Merged text with header '## Introduction' not found"
        assert "This is the first introduction." in merged_text[
            "content"], "Merged content should include first introduction"
        assert "This is another intro text." in merged_text[
            "content"], "Merged content should include second intro text"
        assert len(merged_text["content"].split()
                   ) == 10, "Merged content should have 10 words"
        conclusion_text = next(
            (t for t in result if t["header"] == "## Conclusion"), None)
        assert conclusion_text is not None, "Conclusion text not found"
        assert conclusion_text["content"] == "This is the conclusion.", "Conclusion content should be unchanged"
        captured = capsys.readouterr()
        assert "Deduplicating 3 texts based on headers with agglomerative clustering" in captured.out
        assert "Cluster labels: [0 0 1]" in captured.out
        assert "Merged 2 texts for cluster 0, header: ## Introduction" in captured.out
        assert "Reduced 3 texts to 2 after header-based clustering" in captured.out

    def test_markdown_headers(self, capsys, mock_sentence_transformer):
        texts = [
            {
                "text": "## Introduction\nThis is the first introduction.",
                "doc_index": 0,
                "id": "doc_0",
                "header_level": 2,
                "parent_header": None,
                "header": "## Introduction",
                "content": "This is the first introduction."
            },
            {
                "text": "## Intro\nThis is another intro text.",
                "doc_index": 1,
                "id": "doc_1",
                "header_level": 2,
                "parent_header": None,
                "header": "## Intro",
                "content": "This is another intro text."
            },
            {
                "text": "## Conclusion\nThis is the conclusion.",
                "doc_index": 2,
                "id": "doc_2",
                "header_level": 2,
                "parent_header": None,
                "header": "## Conclusion",
                "content": "This is the conclusion."
            }
        ]
        mock_sentence_transformer.encode.side_effect = lambda headers, **kwargs: np.array([
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0]
        ])
        result = merge_duplicate_texts_agglomerative(
            texts,
            model_name="all-MiniLM-L12-v2",
            device="mps",
            similarity_threshold=0.7,
            batch_size=32
        )
        assert len(
            result) == 2, f"Should deduplicate to 2 texts, got {len(result)}: {[t['header'] for t in result]}"
        merged_text = next(
            (t for t in result if t["header"] == "## Introduction"), None)
        assert merged_text is not None, "Merged text with header '## Introduction' not found"
        assert "This is the first introduction." in merged_text["content"]
        assert "This is another intro text." in merged_text["content"]
        assert len(merged_text["content"].split()) == 10
        conclusion_text = next(
            (t for t in result if t["header"] == "## Conclusion"), None)
        assert conclusion_text is not None, "Conclusion text not found"
        assert conclusion_text["content"] == "This is the conclusion."
        captured = capsys.readouterr()
        assert "Deduplicating 3 texts based on headers with agglomerative clustering" in captured.out
        assert "Cluster labels: [0 0 1]" in captured.out
        assert "Merged 2 texts for cluster 0, header: ## Introduction" in captured.out
        assert "Reduced 3 texts to 2 after header-based clustering" in captured.out

    def test_multiple_markdown_headers(self, capsys, mock_sentence_transformer):
        texts = [
            {
                "text": "## Introduction\nThis is the first introduction.",
                "doc_index": 0,
                "id": "doc_0",
                "header_level": 2,
                "parent_header": None,
                "header": "## Introduction",
                "content": "This is the first introduction."
            },
            {
                "text": "## Intro\nThis is another intro text.",
                "doc_index": 1,
                "id": "doc_1",
                "header_level": 2,
                "parent_header": None,
                "header": "## Intro",
                "content": "This is another intro text."
            },
            {
                "text": "## Overview\nThis is an overview text.",
                "doc_index": 2,
                "id": "doc_2",
                "header_level": 2,
                "parent_header": None,
                "header": "## Overview",
                "content": "This is an overview text."
            },
            {
                "text": "## Conclusion\nThis is the conclusion.",
                "doc_index": 3,
                "id": "doc_3",
                "header_level": 2,
                "parent_header": None,
                "header": "## Conclusion",
                "content": "This is the conclusion."
            }
        ]
        mock_sentence_transformer.encode.side_effect = lambda headers, **kwargs: np.array([
            [1.0, 0.0],
            [0.95, 0.05],
            [0.97, 0.03],
            [0.0, 1.0]
        ])
        result = merge_duplicate_texts_agglomerative(
            texts,
            model_name="all-MiniLM-L12-v2",
            device="mps",
            similarity_threshold=0.7,
            batch_size=32
        )
        assert len(
            result) == 2, f"Should deduplicate to 2 texts, got {len(result)}: {[t['header'] for t in result]}"
        merged_text = next(
            (t for t in result if t["header"] == "## Introduction"), None)
        assert merged_text is not None, "Merged text with header '## Introduction' not found"
        assert "This is the first introduction." in merged_text["content"]
        assert "This is another intro text." in merged_text["content"]
        assert "This is an overview text." in merged_text["content"]
        assert len(merged_text["content"].split()) == 15
        conclusion_text = next(
            (t for t in result if t["header"] == "## Conclusion"), None)
        assert conclusion_text is not None, "Conclusion text not found"
        assert conclusion_text["content"] == "This is the conclusion."
        captured = capsys.readouterr()
        assert "Deduplicating 4 texts based on headers with agglomerative clustering" in captured.out
        assert "Cluster labels: [0 0 0 1]" in captured.out
        assert "Merged 3 texts for cluster 0, header: ## Introduction" in captured.out
        assert "Reduced 4 texts to 2 after header-based clustering" in captured.out

    def test_newline_separation(self, capsys, mock_sentence_transformer):
        texts = [
            {
                "text": "## Introduction\nThis is the first introduction.",
                "doc_index": 0,
                "id": "doc_0",
                "header_level": 2,
                "parent_header": None,
                "header": "## Introduction",
                "content": "This is the first introduction."
            },
            {
                "text": "## Intro\nThis is another intro text.",
                "doc_index": 1,
                "id": "doc_1",
                "header_level": 2,
                "parent_header": None,
                "header": "## Intro",
                "content": "This is another intro text."
            },
            {
                "text": "## Conclusion\nThis is the conclusion.",
                "doc_index": 2,
                "id": "doc_2",
                "header_level": 2,
                "parent_header": None,
                "header": "## Conclusion",
                "content": "This is the conclusion."
            }
        ]
        mock_sentence_transformer.encode.side_effect = lambda headers, **kwargs: np.array([
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0]
        ])
        result = merge_duplicate_texts_agglomerative(
            texts,
            model_name="all-MiniLM-L12-v2",
            device="mps",
            similarity_threshold=0.7,
            batch_size=32
        )
        assert len(
            result) == 2, f"Should deduplicate to 2 texts, got {len(result)}: {[t['header'] for t in result]}"
        merged_text = next(
            (t for t in result if t["header"] == "## Introduction"), None)
        assert merged_text is not None, "Merged text with header '## Introduction' not found"
        assert "\n\n" in merged_text["content"], "Merged content should contain two newlines as separator"
        assert merged_text["content"].count(
            "\n\n") == 1, "Merged content should have exactly one double newline separator"
        assert merged_text["content"].startswith(
            "This is the first introduction.")
        assert merged_text["content"].endswith("This is another intro text.")
        assert len(merged_text["content"].split()) == 10
        conclusion_text = next(
            (t for t in result if t["header"] == "## Conclusion"), None)
        assert conclusion_text is not None, "Conclusion text not found"
        assert conclusion_text["content"] == "This is the conclusion."
        captured = capsys.readouterr()
        assert "Deduplicating 3 texts based on headers with agglomerative clustering" in captured.out
        assert "Cluster labels: [0 0 1]" in captured.out
        assert "Merged 2 texts for cluster 0, header: ## Introduction" in captured.out
        assert "Reduced 3 texts to 2 after header-based clustering" in captured.out


class TestPreprocessTexts:
    def test_basic_preprocessing(self, sample_headers, capsys):
        result = preprocess_texts(
            sample_headers,
            exclude_keywords=[],
            min_header_words=2,
            min_header_level=2,
            parent_keyword=None,
            min_content_words=2
        )
        assert len(result) == 4, f"Expected 4 texts, got {len(result)}"
        assert all(isinstance(t, PreprocessedText) for t in result)
        assert result[0]["id"] == "doc_0"
        assert result[0]["header"] == "## Introduction"
        assert result[0]["content"] == "This is the first introduction."
        assert result[0]["text"].startswith(
            "## Introduction\nThis is the first")
        captured = capsys.readouterr()
        assert "Preprocessing 4 headers" in captured.out
        assert "Preprocessed 4 texts" in captured.out

    def test_exclude_keywords(self, sample_headers, capsys):
        result = preprocess_texts(
            sample_headers,
            exclude_keywords=["introduction"],
            min_header_words=2,
            min_header_level=2,
            parent_keyword=None,
            min_content_words=2
        )
        assert len(
            result) == 2, f"Expected 2 texts after excluding 'introduction', got {len(result)}"
        assert all("introduction" not in t["header"].lower() for t in result)
        captured = capsys.readouterr()
        assert "Excluded: 2 (keywords)" in captured.out

    def test_min_header_level(self, sample_headers, capsys):
        sample_headers[0]["header_level"] = 1
        result = preprocess_texts(
            sample_headers,
            exclude_keywords=[],
            min_header_words=2,
            min_header_level=2,
            parent_keyword=None,
            min_content_words=2
        )
        assert len(
            result) == 3, f"Expected 3 texts after header level filter, got {len(result)}"
        assert all(t["header_level"] >= 2 for t in result)
        captured = capsys.readouterr()
        assert "Excluded: 1 (header level)" in captured.out


class TestEmbedSearch:
    def test_basic_search(self, sample_headers, mock_sentence_transformer, capsys):
        texts = preprocess_texts(sample_headers)
        mock_sentence_transformer.encode.side_effect = [
            np.array([[0.1, 0.2]]),  # Query embedding
            # Chunk embeddings
            np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]),
            np.array([[0.1, 0.2], [0.2, 0.3]])  # Top-k embeddings
        ]
        mock_sentence_transformer.tokenize.return_value = {
            "input_ids": [[1, 2, 3]]}
        with patch("jet.vectors.search_with_mmr.util.cos_sim", return_value=torch.tensor([[0.9, 0.8, 0.7, 0.6]])):
            result = embed_search(
                query="test query",
                texts=texts,
                model_name="all-mpnet-base-v2",
                device="mps",
                top_k=2,
                num_threads=4
            )
        assert len(result) == 2, f"Expected 2 results, got {len(result)}"
        assert all(isinstance(r, SimilarityResult) for r in result)
        assert result[0]["rank"] == 1
        assert result[0]["score"] == 0.9
        assert result[0]["tokens"] == 3
        assert result[0]["embedding"].shape == (2,)
        captured = capsys.readouterr()
        assert "Starting embedding search for 4 texts" in captured.out
        assert "Embedding search returned 2 results" in captured.out

    def test_empty_texts(self, capsys):
        result = embed_search(
            query="test query",
            texts=[],
            model_name="all-mpnet-base-v2",
            device="mps",
            top_k=2
        )
        assert len(result) == 0, "Expected empty results for empty texts"
        captured = capsys.readouterr()
        assert "Starting embedding search for 0 texts" in captured.out


class TestRerankResults:
    def test_basic_reranking(self, sample_headers, mock_cross_encoder, capsys):
        texts = preprocess_texts(sample_headers)
        candidates = [
            {
                "id": t["id"],
                "rank": i + 1,
                "doc_index": t["doc_index"],
                "score": 0.9 - i * 0.1,
                "text": t["text"],
                "tokens": 10,
                "rerank_score": 0.0,
                "diversity_score": 0.0,
                "embedding": np.array([0.1 * i, 0.2 * i]),
                "header_level": t["header_level"],
                "parent_header": t["parent_header"],
                "header": t["header"],
                "content": t["content"]
            } for i, t in enumerate(texts[:2])
        ]
        mock_cross_encoder.predict.return_value = np.array([0.8, 0.9])
        result = rerank_results(
            query="test query",
            candidates=candidates,
            model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
            device="mps",
            batch_size=16
        )
        assert len(result) == 2
        assert result[0]["rerank_score"] == 0.9
        assert result[0]["rank"] == 1
        assert result[1]["rerank_score"] == 0.8
        assert result[1]["rank"] == 2
        captured = capsys.readouterr()
        assert "Reranking 2 candidates" in captured.out
        assert "Reranking completed" in captured.out

    def test_empty_candidates(self, capsys):
        result = rerank_results(
            query="test query",
            candidates=[],
            model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
            device="mps"
        )
        assert len(result) == 0
        captured = capsys.readouterr()
        assert "Reranking 0 candidates" in captured.out


class TestMMRDiversity:
    def test_basic_mmr(self, sample_headers, capsys):
        candidates = [
            {
                "id": f"doc_{i}",
                "rank": i + 1,
                "doc_index": i,
                "score": 0.9 - i * 0.1,
                "text": f"Text {i}",
                "tokens": 10,
                "rerank_score": 0.9 - i * 0.1,
                "diversity_score": 0.0,
                "embedding": np.array([0.1 * i, 0.2 * i]),
                "header_level": 2,
                "parent_header": None,
                "header": f"## Header {i}",
                "content": f"Content {i}"
            } for i in range(3)
        ]
        with patch("jet.vectors.search_with_mmr.util.cos_sim", return_value=torch.tensor([[0.2, 0.1, 0.3]])):
            result = mmr_diversity(
                candidates,
                num_results=2,
                lambda_param=0.5,
                parent_diversity_weight=0.4,
                header_diversity_weight=0.3,
                device="mps"
            )
        assert len(result) == 2
        assert result[0]["rank"] == 1
        assert result[1]["rank"] == 2
        assert result[0]["diversity_score"] > 0
        captured = capsys.readouterr()
        assert "Applying MMR diversity to select 2 results" in captured.out
        assert "MMR diversity selected 2 results" in captured.out

    def test_empty_candidates(self, capsys):
        result = mmr_diversity(
            candidates=[],
            num_results=2,
            lambda_param=0.5,
            parent_diversity_weight=0.4,
            header_diversity_weight=0.3,
            device="mps"
        )
        assert len(result) == 0
        captured = capsys.readouterr()
        assert "Applying MMR diversity to select 2 results" in captured.out


class TestSearchDiverseContext:
    def test_basic_search(self, sample_headers, mock_sentence_transformer, mock_cross_encoder, capsys):
        mock_sentence_transformer.encode.side_effect = [
            np.array([[0.1, 0.2]]),  # Query
            np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]),  # Chunk
            np.array([[0.1, 0.2], [0.2, 0.3]]),  # Top-k
            # Deduplication
            np.array([[1.0, 0.0], [0.95, 0.05], [0.97, 0.03], [0.0, 1.0]])
        ]
        mock_sentence_transformer.tokenize.return_value = {
            "input_ids": [[1, 2, 3]]}
        mock_cross_encoder.predict.return_value = np.array([0.9, 0.8])
        with patch("jet.vectors.search_with_mmr.util.cos_sim", return_value=torch.tensor([[0.9, 0.8, 0.7, 0.6]])):
            result = search_diverse_context(
                query="test query",
                headers=sample_headers,
                model_name="all-mpnet-base-v2",
                rerank_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
                device="mps",
                top_k=2,
                num_results=2
            )
        assert len(result) == 2
        assert all(isinstance(r, SimilarityResult) for r in result)
        assert result[0]["rank"] == 1
        captured = capsys.readouterr()
        assert "Starting search with query" in captured.out
        assert "Search completed" in captured.out

    def test_invalid_inputs(self, sample_headers):
        with pytest.raises(ValueError, match="Query cannot be empty"):
            search_diverse_context("", sample_headers)
        with pytest.raises(ValueError, match="top_k and num_results must be positive"):
            search_diverse_context("test query", sample_headers, top_k=0)
        with pytest.raises(ValueError, match="lambda_param must be between 0 and 1"):
            search_diverse_context(
                "test query", sample_headers, lambda_param=1.5)
        result = search_diverse_context("test query", [])
        assert len(result) == 0
