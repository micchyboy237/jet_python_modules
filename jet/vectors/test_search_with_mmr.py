import pytest
import numpy as np
from unittest.mock import patch
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
            "header": "## Project Introduction",
            "content": "This section introduces the project, outlining its goals and objectives. It provides a comprehensive overview of the initiative.",
            "header_level": 2,
            "parent_header": None
        },
        {
            "header": "### Project Goals",
            "content": "The primary goals include improving efficiency and scalability. We aim to enhance system performance significantly.",
            "header_level": 3,
            "parent_header": "## Project Introduction"
        },
        {
            "header": "## Technical Overview",
            "content": "This section details the technical architecture and implementation strategy. It covers the core components and their interactions.",
            "header_level": 2,
            "parent_header": None
        },
        {
            "header": "### System Architecture",
            "content": "The system is built using a modular architecture. Each module is designed to be independently scalable and maintainable.",
            "header_level": 3,
            "parent_header": "## Technical Overview"
        },
        {
            "header": "## Project Conclusion",
            "content": "This section summarizes the project outcomes and future steps. It reflects on the achievements and lessons learned.",
            "header_level": 2,
            "parent_header": None
        },
        {
            "header": "### Future Steps",
            "content": "Future work includes integrating advanced features and optimizing performance. We plan to expand the system's capabilities.",
            "header_level": 3,
            "parent_header": "## Project Conclusion"
        }
    ]


class TestMergeDuplicateTextsAgglomerative:
    def test_basic_deduplication(self, mock_sentence_transformer):
        texts = [
            {
                "text": "## Project Introduction\nThis section introduces the project, outlining its goals and objectives.",
                "doc_index": 0,
                "id": "doc_0",
                "header_level": 2,
                "parent_header": None,
                "header": "## Project Introduction",
                "content": "This section introduces the project, outlining its goals and objectives."
            },
            {
                "text": "## Project Goals\nThis section outlines the primary objectives of the project.",
                "doc_index": 1,
                "id": "doc_1",
                "header_level": 2,
                "parent_header": None,
                "header": "## Project Goals",
                "content": "This section outlines the primary objectives of the project."
            },
            {
                "text": "## Project Conclusion\nThis section summarizes the project outcomes.",
                "doc_index": 2,
                "id": "doc_2",
                "header_level": 2,
                "parent_header": None,
                "header": "## Project Conclusion",
                "content": "This section summarizes the project outcomes."
            }
        ]
        mock_sentence_transformer.encode.side_effect = lambda headers, **kwargs: np.array([
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0]
        ], dtype=np.float32)
        result = merge_duplicate_texts_agglomerative(
            texts,
            model_name="all-MiniLM-L12-v2",
            device="mps",
            similarity_threshold=0.7,
            batch_size=32
        )
        assert len(result) == 2
        merged_text = next(
            (t for t in result if t["header"] == "## Project Introduction"), None)
        assert merged_text is not None
        assert "This section introduces the project" in merged_text["content"]
        assert "This section outlines the primary objectives" in merged_text["content"]
        assert len(merged_text["content"].split()) == 16
        conclusion_text = next(
            (t for t in result if t["header"] == "## Project Conclusion"), None)
        assert conclusion_text is not None
        assert conclusion_text["content"] == "This section summarizes the project outcomes."

    def test_markdown_headers(self, mock_sentence_transformer):
        texts = [
            {
                "text": "## Project Introduction\nThis section introduces the project, outlining its goals and objectives.",
                "doc_index": 0,
                "id": "doc_0",
                "header_level": 2,
                "parent_header": None,
                "header": "## Project Introduction",
                "content": "This section introduces the project, outlining its goals and objectives."
            },
            {
                "text": "## Project Goals\nThis section outlines the primary objectives of the project.",
                "doc_index": 1,
                "id": "doc_1",
                "header_level": 2,
                "parent_header": None,
                "header": "## Project Goals",
                "content": "This section outlines the primary objectives of the project."
            },
            {
                "text": "## Project Conclusion\nThis section summarizes the project outcomes.",
                "doc_index": 2,
                "id": "doc_2",
                "header_level": 2,
                "parent_header": None,
                "header": "## Project Conclusion",
                "content": "This section summarizes the project outcomes."
            }
        ]
        mock_sentence_transformer.encode.side_effect = lambda headers, **kwargs: np.array([
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0]
        ], dtype=np.float32)
        result = merge_duplicate_texts_agglomerative(
            texts,
            model_name="all-MiniLM-L12-v2",
            device="mps",
            similarity_threshold=0.7,
            batch_size=32
        )
        assert len(result) == 2
        merged_text = next(
            (t for t in result if t["header"] == "## Project Introduction"), None)
        assert merged_text is not None
        assert "This section introduces the project" in merged_text["content"]
        assert "This section outlines the primary objectives" in merged_text["content"]
        assert len(merged_text["content"].split()) == 16
        conclusion_text = next(
            (t for t in result if t["header"] == "## Project Conclusion"), None)
        assert conclusion_text is not None
        assert conclusion_text["content"] == "This section summarizes the project outcomes."

    def test_multiple_markdown_headers(self, mock_sentence_transformer):
        texts = [
            {
                "text": "## Project Introduction\nThis section introduces the project, outlining its goals and objectives.",
                "doc_index": 0,
                "id": "doc_0",
                "header_level": 2,
                "parent_header": None,
                "header": "## Project Introduction",
                "content": "This section introduces the project, outlining its goals and objectives."
            },
            {
                "text": "## Project Goals\nThis section outlines the primary objectives of the project.",
                "doc_index": 1,
                "id": "doc_1",
                "header_level": 2,
                "parent_header": None,
                "header": "## Project Goals",
                "content": "This section outlines the primary objectives of the project."
            },
            {
                "text": "## Technical Overview\nThis section details the technical architecture.",
                "doc_index": 2,
                "id": "doc_2",
                "header_level": 2,
                "parent_header": None,
                "header": "## Technical Overview",
                "content": "This section details the technical architecture."
            },
            {
                "text": "## Project Conclusion\nThis section summarizes the project outcomes.",
                "doc_index": 3,
                "id": "doc_3",
                "header_level": 2,
                "parent_header": None,
                "header": "## Project Conclusion",
                "content": "This section summarizes the project outcomes."
            }
        ]
        mock_sentence_transformer.encode.side_effect = lambda headers, **kwargs: np.array([
            [1.0, 0.0],
            [0.95, 0.05],
            [0.97, 0.03],
            [0.0, 1.0]
        ], dtype=np.float32)
        result = merge_duplicate_texts_agglomerative(
            texts,
            model_name="all-MiniLM-L12-v2",
            device="mps",
            similarity_threshold=0.7,
            batch_size=32
        )
        assert len(result) == 2
        merged_text = next(
            (t for t in result if t["header"] == "## Project Introduction"), None)
        assert merged_text is not None
        assert "This section introduces the project" in merged_text["content"]
        assert "This section outlines the primary objectives" in merged_text["content"]
        assert "This section details the technical architecture" in merged_text["content"]
        assert len(merged_text["content"].split()) == 22
        conclusion_text = next(
            (t for t in result if t["header"] == "## Project Conclusion"), None)
        assert conclusion_text is not None
        assert conclusion_text["content"] == "This section summarizes the project outcomes."

    def test_newline_separation(self, mock_sentence_transformer):
        texts = [
            {
                "text": "## Project Introduction\nThis section introduces the project, outlining its goals and objectives.",
                "doc_index": 0,
                "id": "doc_0",
                "header_level": 2,
                "parent_header": None,
                "header": "## Project Introduction",
                "content": "This section introduces the project, outlining its goals and objectives."
            },
            {
                "text": "## Project Goals\nThis section outlines the primary objectives of the project.",
                "doc_index": 1,
                "id": "doc_1",
                "header_level": 2,
                "parent_header": None,
                "header": "## Project Goals",
                "content": "This section outlines the primary objectives of the project."
            },
            {
                "text": "## Project Conclusion\nThis section summarizes the project outcomes.",
                "doc_index": 2,
                "id": "doc_2",
                "header_level": 2,
                "parent_header": None,
                "header": "## Project Conclusion",
                "content": "This section summarizes the project outcomes."
            }
        ]
        mock_sentence_transformer.encode.side_effect = lambda headers, **kwargs: np.array([
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0]
        ], dtype=np.float32)
        result = merge_duplicate_texts_agglomerative(
            texts,
            model_name="all-MiniLM-L12-v2",
            device="mps",
            similarity_threshold=0.7,
            batch_size=32
        )
        assert len(result) == 2
        merged_text = next(
            (t for t in result if t["header"] == "## Project Introduction"), None)
        assert merged_text is not None
        assert "\n\n" in merged_text["content"]
        assert merged_text["content"].count("\n\n") == 1
        assert merged_text["content"].startswith(
            "This section introduces the project")
        assert merged_text["content"].endswith(
            "This section outlines the primary objectives of the project.")
        assert len(merged_text["content"].split()) == 16
        conclusion_text = next(
            (t for t in result if t["header"] == "## Project Conclusion"), None)
        assert conclusion_text is not None
        assert conclusion_text["content"] == "This section summarizes the project outcomes."


class TestPreprocessTexts:
    def test_basic_preprocessing(self, sample_headers):
        result = preprocess_texts(
            sample_headers,
            exclude_keywords=[],
            min_header_words=2,
            min_header_level=2,
            parent_keyword=None,
            min_content_words=2
        )
        assert len(result) == 6
        assert all(isinstance(t, dict) and all(key in t for key in [
                   "text", "doc_index", "id", "header_level", "parent_header", "header", "content"]) for t in result)
        assert result[0]["id"] == "doc_0"
        assert result[0]["header"] == "## Project Introduction"
        assert result[0]["content"] == "This section introduces the project, outlining its goals and objectives. It provides a comprehensive overview of the initiative."
        assert result[0]["text"].startswith(
            "## Project Introduction\nThis section introduces")

    def test_exclude_keywords(self, sample_headers):
        result = preprocess_texts(
            sample_headers,
            exclude_keywords=["introduction"],
            min_header_words=2,
            min_header_level=2,
            parent_keyword=None,
            min_content_words=2
        )
        assert len(result) == 3
        assert all("introduction" not in t["header"].lower() for t in result)

    def test_min_header_level(self, sample_headers):
        sample_headers[0]["header_level"] = 1
        result = preprocess_texts(
            sample_headers,
            exclude_keywords=[],
            min_header_words=2,
            min_header_level=2,
            parent_keyword=None,
            min_content_words=2
        )
        assert len(result) == 5
        assert all(t["header_level"] >= 2 for t in result)


class TestEmbedSearch:
    def test_basic_search(self, sample_headers, mock_sentence_transformer):
        texts = preprocess_texts(
            sample_headers, min_header_words=2, min_content_words=2)
        mock_sentence_transformer.encode.side_effect = [
            np.array([[0.1, 0.2]], dtype=np.float32),
            np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5],
                     [0.5, 0.6], [0.6, 0.7]], dtype=np.float32),
            np.array([[0.1, 0.2], [0.2, 0.3]], dtype=np.float32)
        ]
        mock_sentence_transformer.tokenize.return_value = {
            "input_ids": [[1, 2, 3]]}
        with patch("jet.vectors.search_with_mmr.util.cos_sim", return_value=torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]], dtype=torch.float32)):
            result = embed_search(
                query="test query",
                texts=texts,
                model_name="all-mpnet-base-v2",
                device="mps",
                top_k=2,
                num_threads=4
            )
        assert len(result) == 2
        assert all(isinstance(r, dict) and all(key in r for key in ["id", "rank", "doc_index", "score", "text", "tokens",
                   "rerank_score", "diversity_score", "embedding", "header_level", "parent_header", "header", "content"]) for r in result)
        assert result[0]["rank"] == 1
        assert result[0]["score"] == pytest.approx(0.9)
        assert result[0]["tokens"] == 3
        assert result[0]["embedding"].shape == (2,)

    def test_empty_texts(self):
        result = embed_search(
            query="test query",
            texts=[],
            model_name="all-mpnet-base-v2",
            device="mps",
            top_k=2
        )
        assert len(result) == 0


class TestRerankResults:
    def test_basic_reranking(self, sample_headers, mock_cross_encoder):
        texts = preprocess_texts(
            sample_headers, min_header_words=2, min_content_words=2)
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
                "embedding": np.array([0.1 * i, 0.2 * i], dtype=np.float32),
                "header_level": t["header_level"],
                "parent_header": t["parent_header"],
                "header": t["header"],
                "content": t["content"]
            } for i, t in enumerate(texts[:2])
        ]
        mock_cross_encoder.predict.side_effect = lambda pairs, **kwargs: np.array(
            []) if not pairs else np.array([-11.3, -11.2], dtype=np.float32)
        result = rerank_results(
            query="test query",
            candidates=candidates,
            model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
            device="mps",
            batch_size=16
        )
        assert len(result) == 2
        assert result[0]["rerank_score"] == -11.2
        assert result[0]["rank"] == 1
        assert result[1]["rerank_score"] == -11.3
        assert result[1]["rank"] == 2

    def test_empty_candidates(self):
        result = rerank_results(
            query="test query",
            candidates=[],
            model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
            device="mps"
        )
        assert len(result) == 0


class TestMMRDiversity:
    def test_basic_mmr(self, sample_headers):
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
                "embedding": np.array([0.1 * i, 0.2 * i], dtype=np.float32),
                "header_level": 2,
                "parent_header": None,
                "header": f"## Header {i}",
                "content": f"Content {i}"
            } for i in range(3)
        ]
        with patch("jet.vectors.search_with_mmr.util.cos_sim", return_value=torch.tensor([[0.2, 0.1, 0.3]], dtype=torch.float32)):
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

    def test_empty_candidates(self):
        result = mmr_diversity(
            candidates=[],
            num_results=2,
            lambda_param=0.5,
            parent_diversity_weight=0.4,
            header_diversity_weight=0.3,
            device="mps"
        )
        assert len(result) == 0


class TestSearchDiverseContext:
    def test_basic_search(self, sample_headers, mock_sentence_transformer, mock_cross_encoder):
        mock_sentence_transformer.encode.side_effect = [
            np.array([[0.1, 0.2]], dtype=np.float32),
            np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5],
                     [0.5, 0.6], [0.6, 0.7]], dtype=np.float32),
            np.array([[0.1, 0.2], [0.2, 0.3]], dtype=np.float32),
            np.array([[1.0, 0.0], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4],
                     [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
        ]
        mock_sentence_transformer.tokenize.return_value = {
            "input_ids": [[1, 2, 3]]}
        mock_cross_encoder.predict.side_effect = lambda pairs, **kwargs: np.array(
            []) if not pairs else np.array([-11.2, -11.3], dtype=np.float32)
        with patch("jet.vectors.search_with_mmr.util.cos_sim", return_value=torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]], dtype=torch.float32)):
            result = search_diverse_context(
                query="project architecture",
                headers=sample_headers,
                model_name="all-mpnet-base-v2",
                rerank_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
                device="mps",
                top_k=2,
                num_results=2,
                min_header_words=2,
                min_content_words=2
            )
        assert len(result) == 2
        assert all(isinstance(r, dict) and all(key in r for key in ["id", "rank", "doc_index", "score", "text", "tokens",
                   "rerank_score", "diversity_score", "embedding", "header_level", "parent_header", "header", "content"]) for r in result)
        assert result[0]["rank"] == 1
        assert result[0]["score"] >= 0.8
        assert result[0]["header"] in [
            "## Technical Overview", "## System Architecture"]

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
