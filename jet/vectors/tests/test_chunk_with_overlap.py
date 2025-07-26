import pytest
import numpy as np
from pathlib import Path
import os
from unittest.mock import patch, Mock
from jet_python_modules.jet.vectors.semantic_search.file_vector_search import (
    get_file_vectors,
    cosine_similarity,
    collect_file_chunks,
    compute_weighted_similarity,
    search_files,
    FileSearchResult,
    DEFAULT_EMBED_MODEL,
    MAX_CONTENT_SIZE
)


@pytest.fixture
def mock_sentence_transformer():
    with patch('jet_python_modules.jet.vectors.semantic_search.file_vector_search.SentenceTransformerRegistry') as mock_registry:
        mock_model = Mock()
        mock_model.encode.side_effect = lambda x, **kwargs: np.array(
            [0.1, 0.2, 0.3]) if isinstance(x, str) else np.array([[0.1, 0.2, 0.3]] * len(x))
        mock_registry.load_model.return_value = mock_model
        yield mock_model


@pytest.fixture
def temp_file(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("This is a test content")
    return str(file_path)


def test_get_file_vectors_valid_file(mock_sentence_transformer, temp_file):
    name_vector, dir_vector, content_vector = get_file_vectors(temp_file)

    assert isinstance(name_vector, np.ndarray)
    assert isinstance(dir_vector, np.ndarray)
    assert isinstance(content_vector, np.ndarray)
    assert name_vector.shape == (3,)
    assert dir_vector.shape == (3,)
    assert content_vector.shape == (3,)
    assert mock_sentence_transformer.encode.call_count == 3


def test_get_file_vectors_non_text_file(mock_sentence_transformer, tmp_path):
    file_path = tmp_path / "test.bin"
    file_path.write_bytes(b"\x00\x01\x02")
    name_vector, dir_vector, content_vector = get_file_vectors(str(file_path))

    assert isinstance(name_vector, np.ndarray)
    assert isinstance(dir_vector, np.ndarray)
    assert content_vector is None
    assert mock_sentence_transformer.encode.call_count == 2


def test_get_file_vectors_nonexistent_file(mock_sentence_transformer):
    with pytest.raises(FileNotFoundError):
        get_file_vectors("nonexistent.txt")


def test_cosine_similarity():
    vec1 = np.array([1.0, 0.0])
    vec2 = np.array([0.0, 1.0])
    sim = cosine_similarity(vec1, vec2)
    assert abs(sim) < 1e-10

    vec1 = np.array([1.0, 1.0])
    vec2 = np.array([1.0, 1.0])
    sim = cosine_similarity(vec1, vec2)
    assert abs(sim - 1.0) < 1e-10

    vec1 = np.array([0.0, 0.0])
    vec2 = np.array([1.0, 1.0])
    sim = cosine_similarity(vec1, vec2)
    assert sim == 0.0


def test_collect_file_chunks_single_file(temp_file):
    file_paths, file_names, parent_dirs, chunks = collect_file_chunks(
        temp_file, extensions={'.txt'})

    assert len(file_paths) == 1
    assert file_paths[0] == temp_file
    assert file_names[0] == 'test.txt'
    assert parent_dirs[0].endswith('tmp') or parent_dirs[0] == 'root'
    assert len(chunks) == 1
    assert chunks[0][0] == temp_file
    assert chunks[0][1] == 'this is a test content'
    assert chunks[0][2] == 0
    assert chunks[0][3] == 21


def test_collect_file_chunks_invalid_path():
    with pytest.raises(ValueError, match="Path nonexistent does not exist"):
        collect_file_chunks("nonexistent")


def test_collect_file_chunks_with_extensions(tmp_path):
    txt_file = tmp_path / "test.txt"
    bin_file = tmp_path / "test.bin"
    txt_file.write_text("text content")
    bin_file.write_bytes(b"binary content")

    file_paths, _, _, _ = collect_file_chunks(
        str(tmp_path), extensions={'.txt'})
    assert len(file_paths) == 1
    assert file_paths[0].endswith('test.txt')


def test_compute_weighted_similarity_with_content():
    query_vec = np.array([1.0, 0.0, 0.0])
    name_vec = np.array([1.0, 0.0, 0.0])
    dir_vec = np.array([0.0, 1.0, 0.0])
    content_vec = np.array([0.0, 0.0, 1.0])

    weighted_sim, name_sim, dir_sim, content_sim = compute_weighted_similarity(
        query_vec, name_vec, dir_vec, content_vec
    )

    assert abs(name_sim - 1.0) < 1e-10
    assert abs(dir_sim) < 1e-10
    assert abs(content_sim) < 1e-10
    assert abs(weighted_sim - (0.5 * 1.0 + 0.3 * 0.0 + 0.2 * 0.0)) < 1e-10


def test_compute_weighted_similarity_no_content():
    query_vec = np.array([1.0, 0.0, 0.0])
    name_vec = np.array([1.0, 0.0, 0.0])
    dir_vec = np.array([0.0, 1.0, 0.0])

    weighted_sim, name_sim, dir_sim, content_sim = compute_weighted_similarity(
        query_vec, name_vec, dir_vec, None
    )

    assert abs(name_sim - 1.0) < 1e-10
    assert abs(dir_sim) < 1e-10
    assert content_sim == 0.0
    assert abs(weighted_sim - (0.5 * 1.0 + 0.3 * 0.0)) < 1e-10


def test_search_files(mock_sentence_transformer, temp_file):
    results = search_files(temp_file, "test query",
                           extensions={'.txt'}, top_k=1)

    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], dict)
    assert results[0]['rank'] == 1
    assert isinstance(results[0]['score'], float)
    assert results[0]['code'] == 'this is a test content'
    assert results[0]['metadata']['file_path'] == temp_file
    assert isinstance(results[0]['metadata']['name_similarity'], float)
    assert isinstance(results[0]['metadata']['dir_similarity'], float)
    assert isinstance(results[0]['metadata']['content_similarity'], float)


def test_search_files_no_results(mock_sentence_transformer, tmp_path):
    results = search_files(str(tmp_path), "test query", extensions={'.bin'})
    assert results == []


def test_search_files_chunking(temp_file):
    # Test with small chunk size to force multiple chunks
    with open(temp_file, 'w') as f:
        f.write("a" * 1000)

    results = search_files(temp_file, "test query",
                           chunk_size=200, chunk_overlap=50)
    assert len(results) > 1
    assert all(r['metadata']['end_idx'] - r['metadata']
               ['start_idx'] <= 200 for r in results)
