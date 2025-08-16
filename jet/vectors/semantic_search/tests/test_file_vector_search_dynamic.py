import os
import tempfile
import shutil
import pytest
import numpy as np
from pathlib import Path
from typing import Callable
from sentence_transformers import SentenceTransformer, CrossEncoder
from nltk.tokenize import word_tokenize
from jet.vectors.semantic_search.file_vector_search_dynamic import (
    cosine_similarity,
    collect_file_chunks,
    compute_dynamic_weights,
    compute_hybrid_similarity,
    merge_results,
    search_files,
    FileSearchResult,
    DEFAULT_EMBED_MODEL,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory with sample files for testing."""
    temp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(temp_dir, "ml_projects"))

    # Sample files
    with open(os.path.join(temp_dir, "ml_projects", "model_training.py"), "w") as f:
        f.write("This is a machine learning training script.")
    with open(os.path.join(temp_dir, "ml_projects", "data_preprocessing.txt"), "w") as f:
        f.write("Data preprocessing for ML models.")
    with open(os.path.join(temp_dir, "notes.txt"), "w") as f:
        f.write("General project notes.")

    yield temp_dir
    shutil.rmtree(temp_dir)


def test_compute_dynamic_weights():
    """Test dynamic weight computation based on query and document characteristics."""
    # Given
    query = "machine learning training"
    file_path = "ml_projects/model_training.py"
    content = "This is a machine learning training script."

    # When
    name_weight, dir_weight, content_weight, metadata_weight = compute_dynamic_weights(
        query, file_path, content)

    # Then
    total = 0.4 + 0.3 + 0.5 + 0.1  # Directory should match 'ml'
    expected_name_weight = 0.4 / total
    expected_dir_weight = 0.3 / total
    expected_content_weight = 0.5 / total
    expected_metadata_weight = 0.1 / total

    assert abs(name_weight - expected_name_weight) < 1e-5
    assert abs(dir_weight - expected_dir_weight) < 1e-5
    assert abs(content_weight - expected_content_weight) < 1e-5
    assert abs(metadata_weight - expected_metadata_weight) < 1e-5


def test_compute_hybrid_similarity(temp_dir):
    """Test hybrid similarity computation with realistic inputs."""
    # Given
    query = "machine learning training"
    file_path = os.path.join(temp_dir, "ml_projects", "model_training.py")
    content = "This is a machine learning training script."
    embed_model = 'all-MiniLM-L6-v2'
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
    tokenized_corpus = [word_tokenize(content.lower())]
    corpus_index = 0
    model = SentenceTransformer(embed_model)
    query_vector = model.encode(query)
    name_vector = model.encode(Path(file_path).name)
    dir_vector = model.encode(Path(file_path).parent.name)
    content_vector = model.encode(content)
    metadata_vector = model.encode(f".py {Path(file_path).name}")

    # When
    weighted_sim, name_sim, dir_sim, content_sim, metadata_sim, cross_encoder_score, bm25_score = compute_hybrid_similarity(
        query, query_vector, name_vector, dir_vector, content_vector, metadata_vector,
        tokenized_corpus, corpus_index, cross_encoder, file_path, content
    )

    # Then
    name_weight, dir_weight, content_weight, metadata_weight = compute_dynamic_weights(
        query, file_path, content)
    expected_weighted_sim = (
        name_weight * name_sim +
        dir_weight * dir_sim +
        content_weight * content_sim +
        metadata_weight * metadata_sim +
        0.2 * cross_encoder_score +
        0.1 * bm25_score
    )
    assert abs(weighted_sim - expected_weighted_sim) < 1e-5
    assert 0 <= name_sim <= 1
    assert 0 <= dir_sim <= 1
    assert 0 <= content_sim <= 1
    assert 0 <= metadata_sim <= 1
    assert 0 <= cross_encoder_score <= 1
    assert 0 <= bm25_score <= 1


def test_search_files(temp_dir):
    """Test file search with hybrid similarity and reranking."""
    # Given
    query = "machine learning training"
    extensions = [".py", ".txt"]

    # When
    results = list(search_files(
        paths=temp_dir,
        query=query,
        extensions=extensions,
        top_k=2,
        embed_model='all-MiniLM-L6-v2',
        chunk_size=100,
        chunk_overlap=20
    ))

    # Then
    expected_files = [
        os.path.join(temp_dir, "ml_projects", "model_training.py"),
        os.path.join(temp_dir, "ml_projects", "data_preprocessing.txt")
    ]
    result_files = [r["metadata"]["file_path"] for r in results]
    assert len(
        results) == 2, f"Expected 2 results, got {len(results)}: {result_files}"
    assert all(
        f in expected_files for f in result_files), f"Expected files {expected_files}, got {result_files}"
    assert all(
        0 <= r["score"] <= 1 for r in results), f"Scores out of range: {[r['score'] for r in results]}"
    assert all(0 <= r["metadata"]["cross_encoder_score"] <=
               1 for r in results), f"Cross-encoder scores out of range: {[r['metadata']['cross_encoder_score'] for r in results]}"
    assert all(0 <= r["metadata"]["bm25_score"] <=
               1 for r in results), f"BM25 scores out of range: {[r['metadata']['bm25_score'] for r in results]}"
