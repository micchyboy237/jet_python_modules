import pytest
from sentence_transformers import SentenceTransformer
from unittest.mock import patch
import numpy as np
from jet.vectors.search_with_mmr import merge_duplicate_texts_agglomerative


def test_merge_duplicate_texts_agglomerative(capsys):
    texts = [
        {
            "text": "Introduction\nThis is the first introduction.",
            "doc_index": 0,
            "id": "doc_0",
            "header_level": 2,
            "parent_header": None,
            "header": "Introduction",
            "content": "This is the first introduction."
        },
        {
            "text": "Intro\nThis is another intro text.",
            "doc_index": 1,
            "id": "doc_1",
            "header_level": 2,
            "parent_header": None,
            "header": "Intro",
            "content": "This is another intro text."
        },
        {
            "text": "Conclusion\nThis is the conclusion.",
            "doc_index": 2,
            "id": "doc_2",
            "header_level": 2,
            "parent_header": None,
            "header": "Conclusion",
            "content": "This is the conclusion."
        }
    ]
    with patch("sentence_transformers.SentenceTransformer") as MockModel:
        mock_model = MockModel.return_value
        mock_model.encode.side_effect = lambda headers, **kwargs: np.array([
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
        (t for t in result if t["header"] == "Introduction"), None)
    assert merged_text is not None, "Merged text with header 'Introduction' not found"
    assert "This is the first introduction." in merged_text[
        "content"], "Merged content should include first introduction"
    assert "This is another intro text." in merged_text[
        "content"], "Merged content should include second intro text"
    assert len(merged_text["content"].split()
               ) == 10, "Merged content should have 10 words"
    conclusion_text = next(
        (t for t in result if t["header"] == "Conclusion"), None)
    assert conclusion_text is not None, "Conclusion text not found"
    assert conclusion_text["content"] == "This is the conclusion.", "Conclusion content should be unchanged"
    captured = capsys.readouterr()
    assert "Deduplicating 3 texts based on headers with agglomerative clustering" in captured.out, "Initial log not found"
    assert "Cluster labels: [0 0 1]" in captured.out, "Cluster labels log not found"
    assert "Merged 2 texts for cluster 0, header: Introduction" in captured.out, "Merge log not found"
    assert "Reduced 3 texts to 2 after header-based clustering" in captured.out, "Final log not found"


def test_merge_duplicate_texts_agglomerative_with_markdown(capsys):
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
    with patch("sentence_transformers.SentenceTransformer") as MockModel:
        mock_model = MockModel.return_value
        mock_model.encode.side_effect = lambda headers, **kwargs: np.array([
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
    assert "Deduplicating 3 texts based on headers with agglomerative clustering" in captured.out, "Initial log not found"
    assert "Cluster labels: [0 0 1]" in captured.out, "Cluster labels log not found"
    assert "Merged 2 texts for cluster 0, header: ## Introduction" in captured.out, "Merge log not found"
    assert "Reduced 3 texts to 2 after header-based clustering" in captured.out, "Final log not found"


def test_merge_duplicate_texts_agglomerative_multiple_markdown_headers(capsys):
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
    with patch("sentence_transformers.SentenceTransformer") as MockModel:
        mock_model = MockModel.return_value
        mock_model.encode.side_effect = lambda headers, **kwargs: np.array([
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
    assert "This is the first introduction." in merged_text[
        "content"], "Merged content should include first introduction"
    assert "This is another intro text." in merged_text[
        "content"], "Merged content should include intro text"
    assert "This is an overview text." in merged_text[
        "content"], "Merged content should include overview text"
    assert len(merged_text["content"].split()
               ) == 15, "Merged content should have 15 words"
    conclusion_text = next(
        (t for t in result if t["header"] == "## Conclusion"), None)
    assert conclusion_text is not None, "Conclusion text not found"
    assert conclusion_text["content"] == "This is the conclusion.", "Conclusion content should be unchanged"
    captured = capsys.readouterr()
    assert "Deduplicating 4 texts based on headers with agglomerative clustering" in captured.out, "Initial log not found"
    assert "Cluster labels: [0 0 0 1]" in captured.out, "Cluster labels log not found"
    assert "Merged 3 texts for cluster 0, header: ## Introduction" in captured.out, "Merge log not found"
    assert "Reduced 4 texts to 2 after header-based clustering" in captured.out, "Final log not found"


def test_merge_duplicate_texts_agglomerative_newline_separation(capsys):
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
    with patch("sentence_transformers.SentenceTransformer") as MockModel:
        mock_model = MockModel.return_value
        mock_model.encode.side_effect = lambda headers, **kwargs: np.array([
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
        "This is the first introduction."), "Merged content should start with first introduction"
    assert merged_text["content"].endswith(
        "This is another intro text."), "Merged content should end with second intro text"
    assert len(merged_text["content"].split()
               ) == 10, "Merged content should have 10 words"
    conclusion_text = next(
        (t for t in result if t["header"] == "## Conclusion"), None)
    assert conclusion_text is not None, "Conclusion text not found"
    assert conclusion_text["content"] == "This is the conclusion.", "Conclusion content should be unchanged"
    captured = capsys.readouterr()
    assert "Deduplicating 3 texts based on headers with agglomerative clustering" in captured.out, "Initial log not found"
    assert "Cluster labels: [0 0 1]" in captured.out, "Cluster labels log not found"
    assert "Merged 2 texts for cluster 0, header: ## Introduction" in captured.out, "Merge log not found"
    assert "Reduced 3 texts to 2 after header-based clustering" in captured.out, "Final log not found"
