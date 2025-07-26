import re
import pytest
import numpy as np
from pathlib import Path
import os
from unittest.mock import patch, Mock
from jet.vectors.semantic_search.file_vector_search import (
    cosine_similarity,
    collect_file_chunks,
    compute_weighted_similarity,
    merge_results,
    search_files,
    FileSearchResult,
    DEFAULT_EMBED_MODEL,
    MAX_CONTENT_SIZE
)
import logging


@pytest.fixture
def mock_sentence_transformer():
    with patch('jet.vectors.semantic_search.file_vector_search.SentenceTransformerRegistry') as mock_registry:
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
    """
    Given: A temporary text file with known content
    When: Collecting chunks with specific extensions and optional tokenizer
    Then: Returns expected file metadata and chunk data with correct token counts
    """
    def custom_tokenizer(text): return len(text.split())
    expected_content = "this is a test content"
    expected_file_path = temp_file
    expected_file_name = 'test.txt'
    expected_parent_dir = Path(temp_file).parent.name.lower()
    # 5 tokens: "this", "is", "a", "test", "content"
    expected_chunk = (expected_file_path, expected_content, 0, 22, 5)
    # Same for space-based tokenizer
    expected_chunk_custom = (expected_file_path, expected_content, 0, 22, 5)

    # Test with default tokenizer
    file_paths, file_names, parent_dirs, chunks = collect_file_chunks(
        temp_file, extensions=['.txt'])
    assert len(file_paths) == 1, "Expected exactly one file path"
    assert file_paths[0] == expected_file_path, f"Expected file path {expected_file_path}, got {file_paths[0]}"
    assert file_names[0] == expected_file_name, f"Expected file name {expected_file_name}, got {file_names[0]}"
    assert parent_dirs[
        0] == expected_parent_dir, f"Expected parent directory {expected_parent_dir}, got {parent_dirs[0]}"
    assert len(chunks) == 1, "Expected exactly one chunk"
    assert chunks[0][0] == expected_chunk[
        0], f"Expected chunk file path {expected_chunk[0]}, got {chunks[0][0]}"
    assert chunks[0][1] == expected_chunk[
        1], f"Expected chunk content {expected_chunk[1]}, got {chunks[0][1]}"
    assert chunks[0][2] == expected_chunk[
        2], f"Expected chunk start index {expected_chunk[2]}, got {chunks[0][2]}"
    assert chunks[0][3] == expected_chunk[
        3], f"Expected chunk end index {expected_chunk[3]}, got {chunks[0][3]}"
    assert chunks[0][4] == expected_chunk[
        4], f"Expected chunk num_tokens {expected_chunk[4]}, got {chunks[0][4]}"

    # Test with custom tokenizer
    file_paths, file_names, parent_dirs, chunks = collect_file_chunks(
        temp_file, extensions=['.txt'], tokenizer=custom_tokenizer)
    assert chunks[0][4] == expected_chunk_custom[
        4], f"Expected chunk num_tokens {expected_chunk_custom[4]} with custom tokenizer, got {chunks[0][4]}"


def test_collect_file_chunks_invalid_path():
    with pytest.raises(ValueError, match="Path nonexistent does not exist"):
        collect_file_chunks("nonexistent")


def test_collect_file_chunks_with_extensions(tmp_path):
    txt_file = tmp_path / "test.txt"
    bin_file = tmp_path / "test.bin"
    txt_file.write_text("text content")
    bin_file.write_bytes(b"binary content")
    file_paths, _, _, _ = collect_file_chunks(
        str(tmp_path), extensions=['.txt'])
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
    assert abs(weighted_sim - (0.4 * 1.0 + 0.2 * 0.0 + 0.4 * 0.0)) < 1e-10


def test_compute_weighted_similarity_no_content():
    query_vec = np.array([1.0, 0.0, 0.0])
    name_vec = np.array([1.0, 0.0, 0.0])
    dir_vec = np.array([0.0, 1.0, 0.0])
    content_vec = None
    weighted_sim, name_sim, dir_sim, content_sim = compute_weighted_similarity(
        query_vec, name_vec, dir_vec, content_vec
    )
    assert abs(name_sim - 1.0) < 1e-10
    assert abs(dir_sim) < 1e-10
    assert abs(content_sim) < 1e-10
    assert abs(weighted_sim - (0.4 * 1.0 + 0.2 * 0.0)) < 1e-10


def test_search_files(mock_sentence_transformer, temp_file):
    """
    Given: A temporary text file with known content
    When: Searching with a query, specific extensions, optional tokenizer, and split_chunks
    Then: Returns a list of up to top_k results with expected structure and token count
    """
    query = "test query"
    expected_content = "this is a test content"
    expected_file_path = temp_file
    top_k = 1
    def custom_tokenizer(text): return len(
        text.split())  # Simple space-based tokenizer
    expected_num_tokens_default = 5  # "this", "is", "a", "test", "content"
    expected_num_tokens_custom = 5   # Same for this content with space-based tokenizer

    # Test with default tokenizer, merging chunks (split_chunks=False)
    results = list(search_files(temp_file, query,
                   extensions=['.txt'], top_k=top_k))
    assert isinstance(results, list)
    assert len(
        results) <= top_k, f"Expected at most {top_k} results, got {len(results)}"
    assert len(results) == 1, "Expected exactly one result"
    assert isinstance(results[0], dict)
    assert results[0]['rank'] == 1
    assert isinstance(results[0]['score'], float)
    assert results[0]['code'] == expected_content
    assert results[0]['metadata']['file_path'] == expected_file_path
    assert results[0]['metadata']['chunk_idx'] == 0, "Expected chunk_idx to be 0 for single chunk"
    assert isinstance(results[0]['metadata']['name_similarity'], float)
    assert isinstance(results[0]['metadata']['dir_similarity'], float)
    assert isinstance(results[0]['metadata']['content_similarity'], float)
    assert results[0]['metadata'][
        'num_tokens'] == expected_num_tokens_default, f"Expected {expected_num_tokens_default} tokens, got {results[0]['metadata']['num_tokens']}"

    # Test with custom tokenizer, merging chunks (split_chunks=False)
    results = list(search_files(temp_file, query, extensions=[
                   '.txt'], top_k=top_k, tokenizer=custom_tokenizer))
    assert results[0]['metadata'][
        'num_tokens'] == expected_num_tokens_custom, f"Expected {expected_num_tokens_custom} tokens with custom tokenizer, got {results[0]['metadata']['num_tokens']}"

    # Test with split_chunks=True (single chunk, so same result)
    results = list(search_files(temp_file, query, extensions=[
                   '.txt'], top_k=top_k, split_chunks=True))
    assert len(results) == 1
    assert results[0]['code'] == expected_content
    assert results[0]['metadata']['num_tokens'] == expected_num_tokens_default


def test_search_files_no_results(mock_sentence_transformer, tmp_path):
    """
    Given: An empty directory with no matching files
    When: Searching with a query and specific extensions
    Then: Returns an empty list
    """
    logging.debug("Starting test_search_files_no_results")
    results = list(search_files(
        str(tmp_path), "test query", extensions=['.bin']))
    logging.debug(f"Collected results: {results}")

    assert results == []


def test_search_files_chunking(temp_file):
    """
    Given: A temporary text file with content exceeding chunk size
    When: Searching with a query, chunk parameters, optional tokenizer, and split_chunks
    Then: Returns chunked or merged results with correct sizes, indices, and token counts
    """
    content = "word " * 200  # 200 words + 199 spaces
    with open(temp_file, 'w') as f:
        f.write(content)

    def custom_tokenizer(text): return len(text.split())
    chunk_size = 200
    chunk_overlap = 50

    # Test with default tokenizer, merging chunks (split_chunks=False)
    results_default = list(search_files(
        temp_file, "test query", chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    assert len(results_default) == 1, "Expected one merged result"
    assert results_default[0]['code'] == content, "Expected full content after merging"
    expected_tokens_merged = len(re.findall(r'\b\w+\b|[^\w\s]', content))
    assert results_default[0]['metadata'][
        'num_tokens'] == expected_tokens_merged, f"Expected {expected_tokens_merged} tokens, got {results_default[0]['metadata']['num_tokens']}"
    assert results_default[0]['metadata']['chunk_idx'] == 0, "Expected chunk_idx to be 0 for merged result"

    # Test with custom tokenizer, merging chunks (split_chunks=False)
    results_custom = list(search_files(temp_file, "test query", chunk_size=chunk_size,
                          chunk_overlap=chunk_overlap, tokenizer=custom_tokenizer))
    expected_tokens_merged_custom = len(content.split())
    assert results_custom[0]['metadata'][
        'num_tokens'] == expected_tokens_merged_custom, f"Expected {expected_tokens_merged_custom} tokens with custom tokenizer, got {results_custom[0]['metadata']['num_tokens']}"

    # Test with split_chunks=True
    results_split = list(search_files(temp_file, "test query",
                         chunk_size=chunk_size, chunk_overlap=chunk_overlap, split_chunks=True))
    assert len(results_split) > 1
    assert all(r['metadata']['end_idx'] - r['metadata']
               ['start_idx'] <= chunk_size for r in results_split)
    assert all(isinstance(r['metadata']['chunk_idx'], int)
               for r in results_split), "All chunk_idx should be integers"
    assert [r['metadata']['chunk_idx'] for r in results_split] == list(range(len(
        results_split))), f"Expected chunk indices 0 to {len(results_split)-1}, got {[r['metadata']['chunk_idx'] for r in results_split]}"
    for r in results_split:
        expected_tokens = len(re.findall(r'\b\w+\b|[^\w\s]', r['code']))
        assert r['metadata']['num_tokens'] == expected_tokens, f"Expected {expected_tokens} tokens, got {r['metadata']['num_tokens']}"


def test_search_files_with_threshold_and_yielding(mock_sentence_transformer, temp_file):
    """
    Given: A temporary text file with known content
    When: Searching with a query, threshold, top_k, optional tokenizer, and split_chunks
    Then: Yields up to top_k results above threshold with expected structure and token count
    """
    query = "test query"
    expected_threshold = 0.1
    expected_content = "this is a test content"
    expected_file_path = temp_file
    top_k = 1
    def custom_tokenizer(text): return len(text.split())
    expected_num_tokens_default = 5  # "this", "is", "a", "test", "content"
    expected_num_tokens_custom = 5

    # Test with default tokenizer, merging chunks (split_chunks=False)
    results = []
    for result in search_files(temp_file, query, extensions=['.txt'], top_k=top_k, threshold=expected_threshold):
        results.append(result)
    assert len(
        results) <= top_k, f"Expected at most {top_k} results, got {len(results)}"
    assert len(results) == 1, "Expected exactly one result"
    assert isinstance(results[0], dict), "Result should be a dictionary"
    assert results[0]['rank'] == 1, "Rank should be 1 after sorting"
    assert results[0]['score'] >= expected_threshold, f"Score {results[0]['score']} should meet threshold {expected_threshold}"
    assert results[0]['code'] == expected_content, f"Expected content {expected_content}, got {results[0]['code']}"
    assert results[0]['metadata'][
        'file_path'] == expected_file_path, f"Expected file path {expected_file_path}, got {results[0]['metadata']['file_path']}"
    assert results[0]['metadata']['chunk_idx'] == 0, "Expected chunk_idx to be 0 for single chunk"
    assert isinstance(results[0]['metadata']['name_similarity'],
                      float), "Name similarity should be float"
    assert isinstance(results[0]['metadata']['dir_similarity'],
                      float), "Dir similarity should be float"
    assert isinstance(results[0]['metadata']['content_similarity'],
                      float), "Content similarity should be float"
    assert results[0]['metadata'][
        'num_tokens'] == expected_num_tokens_default, f"Expected {expected_num_tokens_default} tokens, got {results[0]['metadata']['num_tokens']}"

    # Test with custom tokenizer, merging chunks (split_chunks=False)
    results = []
    for result in search_files(temp_file, query, extensions=['.txt'], top_k=top_k, threshold=expected_threshold, tokenizer=custom_tokenizer):
        results.append(result)
    assert results[0]['metadata'][
        'num_tokens'] == expected_num_tokens_custom, f"Expected {expected_num_tokens_custom} tokens with custom tokenizer, got {results[0]['metadata']['num_tokens']}"

    # Test with split_chunks=True (single chunk, so same result)
    results = []
    for result in search_files(temp_file, query, extensions=['.txt'], top_k=top_k, threshold=expected_threshold, split_chunks=True):
        results.append(result)
    assert len(results) == 1
    assert results[0]['code'] == expected_content
    assert results[0]['metadata']['num_tokens'] == expected_num_tokens_default


class TestMergeResults:
    def test_merge_adjacent_chunks(self, tmp_path):
        """
        Given: Multiple adjacent chunks from the same file
        When: Merging results with optional tokenizer
        Then: Combines into a single result with correct content, metadata, and token count
        """
        file_path = str(tmp_path / "test.txt")
        content = "first chunk second chunk"
        with open(file_path, 'w') as f:
            f.write(content)

        def custom_tokenizer(text): return len(text.split())
        results = [
            {
                "rank": 1,
                "score": 0.8,
                "code": "first chunk ",
                "metadata": {
                    "file_path": file_path,
                    "start_idx": 0,
                    "end_idx": 12,
                    "chunk_idx": 0,
                    "name_similarity": 0.7,
                    "dir_similarity": 0.6,
                    "content_similarity": 0.9,
                    "num_tokens": 2
                }
            },
            {
                "rank": 2,
                "score": 0.7,
                "code": "second chunk",
                "metadata": {
                    "file_path": file_path,
                    "start_idx": 12,
                    "end_idx": 23,
                    "chunk_idx": 1,
                    "name_similarity": 0.7,
                    "dir_similarity": 0.6,
                    "content_similarity": 0.8,
                    "num_tokens": 2
                }
            }
        ]
        expected_default = [
            {
                "rank": 1,
                "score": 0.75,
                "code": "first chunk second chunk",
                "metadata": {
                    "file_path": file_path,
                    "start_idx": 0,
                    "end_idx": 23,
                    "chunk_idx": 0,
                    "name_similarity": 0.7,
                    "dir_similarity": 0.6,
                    "content_similarity": 0.85,
                    "num_tokens": 4  # "first", "chunk", "second", "chunk"
                }
            }
        ]
        expected_custom = [
            {
                "rank": 1,
                "score": 0.75,
                "code": "first chunk second chunk",
                "metadata": {
                    "file_path": file_path,
                    "start_idx": 0,
                    "end_idx": 23,
                    "chunk_idx": 0,
                    "name_similarity": 0.7,
                    "dir_similarity": 0.6,
                    "content_similarity": 0.85,
                    # "first", "chunk", "second", "chunk" (space-based)
                    "num_tokens": 4
                }
            }
        ]
        # Test with default tokenizer
        merged = merge_results(results)
        assert len(merged) == 1
        assert merged[0]["code"] == expected_default[0]["code"]
        assert merged[0]["metadata"]["start_idx"] == expected_default[0]["metadata"]["start_idx"]
        assert merged[0]["metadata"]["end_idx"] == expected_default[0]["metadata"]["end_idx"]
        assert merged[0]["metadata"]["chunk_idx"] == expected_default[0]["metadata"]["chunk_idx"]
        assert abs(merged[0]["score"] - expected_default[0]["score"]) < 1e-10
        assert abs(merged[0]["metadata"]["content_similarity"] -
                   expected_default[0]["metadata"]["content_similarity"]) < 1e-10
        assert merged[0]["metadata"]["name_similarity"] == expected_default[0]["metadata"]["name_similarity"]
        assert merged[0]["metadata"]["dir_similarity"] == expected_default[0]["metadata"]["dir_similarity"]
        assert merged[0]["metadata"]["num_tokens"] == expected_default[0]["metadata"]["num_tokens"]
        # Test with custom tokenizer
        merged = merge_results(results, tokenizer=custom_tokenizer)
        assert merged[0]["metadata"]["num_tokens"] == expected_custom[0]["metadata"]["num_tokens"]

    def test_non_adjacent_chunks(self, tmp_path):
        """
        Given: Non-adjacent chunks from the same file
        When: Merging results
        Then: Keeps chunks separate with correct metadata
        """
        file_path = str(tmp_path / "test.txt")
        results = [
            {
                "rank": 1,
                "score": 0.8,
                "code": "first chunk",
                "metadata": {
                    "file_path": file_path,
                    "start_idx": 0,
                    "end_idx": 11,
                    "chunk_idx": 0,
                    "name_similarity": 0.7,
                    "dir_similarity": 0.6,
                    "content_similarity": 0.9
                }
            },
            {
                "rank": 2,
                "score": 0.7,
                "code": "second chunk",
                "metadata": {
                    "file_path": file_path,
                    "start_idx": 20,
                    "end_idx": 31,
                    "chunk_idx": 1,
                    "name_similarity": 0.7,
                    "dir_similarity": 0.6,
                    "content_similarity": 0.8
                }
            }
        ]
        expected = [
            {
                "rank": 1,
                "score": 0.8,
                "code": "first chunk",
                "metadata": {
                    "file_path": file_path,
                    "start_idx": 0,
                    "end_idx": 11,
                    "chunk_idx": 0,
                    "name_similarity": 0.7,
                    "dir_similarity": 0.6,
                    "content_similarity": 0.9
                }
            },
            {
                "rank": 2,
                "score": 0.7,
                "code": "second chunk",
                "metadata": {
                    "file_path": file_path,
                    "start_idx": 20,
                    "end_idx": 31,
                    "chunk_idx": 0,
                    "name_similarity": 0.7,
                    "dir_similarity": 0.6,
                    "content_similarity": 0.8
                }
            }
        ]
        merged = merge_results(results)
        assert len(merged) == 2
        for i in range(2):
            assert merged[i]["code"] == expected[i]["code"]
            assert merged[i]["metadata"]["start_idx"] == expected[i]["metadata"]["start_idx"]
            assert merged[i]["metadata"]["end_idx"] == expected[i]["metadata"]["end_idx"]
            assert merged[i]["metadata"]["chunk_idx"] == expected[i]["metadata"]["chunk_idx"]
            assert merged[i]["score"] == expected[i]["score"]
            assert merged[i]["metadata"]["content_similarity"] == expected[i]["metadata"]["content_similarity"]

    def test_multiple_files(self, tmp_path):
        """
        Given: Chunks from different files
        When: Merging results
        Then: Groups by file and merges adjacent chunks correctly
        """
        file1 = str(tmp_path / "file1.txt")
        file2 = str(tmp_path / "file2.txt")
        results = [
            {
                "rank": 1,
                "score": 0.8,
                "code": "file1 chunk1",
                "metadata": {
                    "file_path": file1,
                    "start_idx": 0,
                    "end_idx": 12,
                    "chunk_idx": 0,
                    "name_similarity": 0.7,
                    "dir_similarity": 0.6,
                    "content_similarity": 0.9
                }
            },
            {
                "rank": 2,
                "score": 0.7,
                "code": "file2 chunk1",
                "metadata": {
                    "file_path": file2,
                    "start_idx": 0,
                    "end_idx": 12,
                    "chunk_idx": 0,
                    "name_similarity": 0.6,
                    "dir_similarity": 0.5,
                    "content_similarity": 0.8
                }
            }
        ]
        expected = [
            {
                "rank": 1,
                "score": 0.8,
                "code": "file1 chunk1",
                "metadata": {
                    "file_path": file1,
                    "start_idx": 0,
                    "end_idx": 12,
                    "chunk_idx": 0,
                    "name_similarity": 0.7,
                    "dir_similarity": 0.6,
                    "content_similarity": 0.9
                }
            },
            {
                "rank": 2,
                "score": 0.7,
                "code": "file2 chunk1",
                "metadata": {
                    "file_path": file2,
                    "start_idx": 0,
                    "end_idx": 12,
                    "chunk_idx": 0,
                    "name_similarity": 0.6,
                    "dir_similarity": 0.5,
                    "content_similarity": 0.8
                }
            }
        ]
        merged = merge_results(results)
        assert len(merged) == 2
        for i in range(2):
            assert merged[i]["code"] == expected[i]["code"]
            assert merged[i]["metadata"]["file_path"] == expected[i]["metadata"]["file_path"]
            assert merged[i]["score"] == expected[i]["score"]

    def test_empty_results(self):
        """
        Given: An empty list of results
        When: Merging results
        Then: Returns an empty list
        """
        results = []
        expected = []
        merged = merge_results(results)
        assert merged == expected
