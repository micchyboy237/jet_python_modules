from typing import List, Set, Union, Tuple, Optional
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from jet.vectors.semantic_search.file_vector_search import (
    search_files,
    get_file_vectors,
    cosine_similarity,
    collect_file_chunks,
    compute_weighted_similarity,
    FileSearchResult,
    FileSearchMetadata
)
from jet.models.model_types import EmbedModelType


@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup fixture to reset mocks after each test."""
    yield
    # Reset mocks to ensure clean state
    patch.stopall()


@pytest.fixture
def mock_sentence_transformer():
    """Fixture to mock SentenceTransformerRegistry and SentenceTransformer."""
    with patch("jet.vectors.semantic_search.file_vector_search.SentenceTransformerRegistry") as mock_registry:
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda x, convert_to_numpy=True, batch_size=32: (
            np.array([0.1, 0.2, 0.3]) if isinstance(x, str) and "query" in x.lower()
            else np.array([0.4, 0.5, 0.6]) if isinstance(x, str) and "file1" in x.lower()
            else np.array([0.2, 0.3, 0.4]) if isinstance(x, str) and "file2" in x.lower()
            else np.array([0.3, 0.4, 0.5]) if isinstance(x, str) and "test_dir" in x.lower()
            else np.array([0.5, 0.6, 0.7]) if isinstance(x, str) and "content" in x.lower()
            else np.array([[0.4, 0.5, 0.6]] * len(x)) if isinstance(x, list)
            else np.array([0.4, 0.5, 0.6])
        )
        mock_registry.load_model.return_value = mock_model
        yield mock_model


@pytest.fixture
def mock_os_walk():
    """Fixture to mock os.walk for directory traversal."""
    with patch("jet.vectors.semantic_search.file_vector_search.os.walk") as mock_walk:
        yield mock_walk


@pytest.fixture
def mock_os_path():
    """Fixture to mock os.path functions."""
    with patch("jet.vectors.semantic_search.file_vector_search.os.path") as mock_path:
        yield mock_path


@pytest.fixture
def mock_open():
    """Fixture to mock open function for file content reading."""
    with patch("builtins.open", new_callable=MagicMock) as mock_file:
        yield mock_file


@pytest.fixture
def mock_pathlib_path():
    """Fixture to mock Path objects."""
    with patch("jet.vectors.semantic_search.file_vector_search.Path") as mock_path:
        mock_path_obj = MagicMock()
        mock_path_obj.name = "file1.txt"
        mock_path_obj.suffix = ".txt"
        mock_path_obj.parent.name = "test"
        # Add relative_to implementation to return a consistent relative path
        mock_path_obj.relative_to = MagicMock(
            return_value=Path("dir/file1.txt"))
        mock_path.return_value = mock_path_obj
        yield mock_path


class TestSearchFiles:
    """Tests for the search_files function."""

    def test_search_single_directory(self, mock_sentence_transformer, mock_os_walk, mock_os_path, mock_open):
        """Test searching a single directory with multiple files."""
        # Given: A directory with two files and a query
        directory = "/test/dir"
        query = "test query"
        extensions = {".txt"}
        expected_results = [
            {
                "rank": 1,
                "score": 0.39,  # 0.5*0.4 + 0.3*0.3 + 0.2*0.5
                "code": "sample content",
                "metadata": {
                    "file_path": "/test/dir/file1.txt",
                    "start_idx": 0,
                    "end_idx": 13,
                    "name_similarity": 0.4,
                    "dir_similarity": 0.3,
                    "content_similarity": 0.5
                }
            },
            {
                "rank": 2,
                "score": 0.29,  # 0.5*0.2 + 0.3*0.3 + 0.2*0.5
                "code": "sample content",
                "metadata": {
                    "file_path": "/test/dir/file2.txt",
                    "start_idx": 0,
                    "end_idx": 13,
                    "name_similarity": 0.2,
                    "dir_similarity": 0.3,
                    "content_similarity": 0.5
                }
            }
        ]
        mock_os_path.exists.return_value = True
        mock_os_path.isfile.return_value = False
        mock_os_path.isdir.return_value = True
        mock_os_walk.return_value = [
            ("/test/dir", [], ["file1.txt", "file2.txt"])]
        mock_open.return_value.__enter__.return_value.read.return_value = "sample content"

        # When: Searching the directory
        results = search_files(directory, query, extensions, top_k=2)

        # Then: Expect correct file paths and similarity scores
        assert results == expected_results
        mock_sentence_transformer.encode.assert_called()
        mock_os_walk.assert_called_once_with(directory)
        mock_open.assert_called()

    def test_search_single_file(self, mock_sentence_transformer, mock_os_path, mock_open):
        """Test searching a single file path."""
        # Given: A single file path and a query
        file_path = "/test/file1.txt"
        query = "test query"
        extensions = {".txt"}
        expected_results = [{
            "rank": 1,
            "score": 0.39,  # 0.5*0.4 + 0.3*0.3 + 0.2*0.5
            "code": "sample content",
            "metadata": {
                "file_path": file_path,
                "start_idx": 0,
                "end_idx": 13,
                "name_similarity": 0.4,
                "dir_similarity": 0.3,
                "content_similarity": 0.5
            }
        }]
        mock_os_path.exists.return_value = True
        mock_os_path.isfile.return_value = True
        mock_os_path.isdir.return_value = False
        mock_open.return_value.__enter__.return_value.read.return_value = "sample content"

        # When: Searching the single file
        results = search_files(file_path, query, extensions, top_k=2)

        # Then: Expect the file path with correct similarity score
        assert results == expected_results
        mock_sentence_transformer.encode.assert_called()
        mock_open.assert_called_with(file_path, 'r', encoding='utf-8')

    def test_search_multiple_paths(self, mock_sentence_transformer, mock_os_walk, mock_os_path, mock_open):
        """Test searching multiple paths (directory and file)."""
        # Given: A directory and a file path with a query
        paths = ["/test/dir", "/test/file3.txt"]
        query = "test query"
        extensions = {".txt"}
        expected_results = [
            {
                "rank": 1,
                "score": 0.39,  # 0.5*0.4 + 0.3*0.3 + 0.2*0.5
                "code": "sample content",
                "metadata": {
                    "file_path": "/test/dir/file1.txt",
                    "start_idx": 0,
                    "end_idx": 13,
                    "name_similarity": 0.4,
                    "dir_similarity": 0.3,
                    "content_similarity": 0.5
                }
            },
            {
                "rank": 2,
                "score": 0.39,  # 0.5*0.4 + 0.3*0.3 + 0.2*0.5
                "code": "sample content",
                "metadata": {
                    "file_path": "/test/file3.txt",
                    "start_idx": 0,
                    "end_idx": 13,
                    "name_similarity": 0.4,
                    "dir_similarity": 0.3,
                    "content_similarity": 0.5
                }
            },
            {
                "rank": 3,
                "score": 0.29,  # 0.5*0.2 + 0.3*0.3 + 0.2*0.5
                "code": "sample content",
                "metadata": {
                    "file_path": "/test/dir/file2.txt",
                    "start_idx": 0,
                    "end_idx": 13,
                    "name_similarity": 0.2,
                    "dir_similarity": 0.3,
                    "content_similarity": 0.5
                }
            }
        ]
        mock_os_path.exists.side_effect = [True, True]
        mock_os_path.isfile.side_effect = [False, True]
        mock_os_path.isdir.side_effect = [True, False]
        mock_os_walk.return_value = [
            ("/test/dir", [], ["file1.txt", "file2.txt"])]
        mock_open.return_value.__enter__.return_value.read.return_value = "sample content"

        # When: Searching multiple paths
        results = search_files(paths, query, extensions, top_k=3)

        # Then: Expect all files from both paths with correct scores
        assert results == expected_results
        mock_sentence_transformer.encode.assert_called()
        mock_os_walk.assert_called_once_with("/test/dir")
        mock_open.assert_called()

    def test_search_with_extension_filter(self, mock_sentence_transformer, mock_os_walk, mock_os_path, mock_open):
        """Test searching with extension filtering."""
        # Given: A directory with mixed file types and an extension filter
        directory = "/test/dir"
        query = "test query"
        extensions = {".txt"}
        expected_results = [{
            "rank": 1,
            "score": 0.39,  # 0.5*0.4 + 0.3*0.3 + 0.2*0.5
            "code": "sample content",
            "metadata": {
                "file_path": "/test/dir/file1.txt",
                "start_idx": 0,
                "end_idx": 13,
                "name_similarity": 0.4,
                "dir_similarity": 0.3,
                "content_similarity": 0.5
            }
        }]
        mock_os_path.exists.return_value = True
        mock_os_path.isfile.return_value = False
        mock_os_path.isdir.return_value = True
        mock_os_walk.return_value = [
            ("/test/dir", [], ["file1.txt", "file2.py"])]
        mock_open.return_value.__enter__.return_value.read.return_value = "sample content"

        # When: Searching with extension filter
        results = search_files(directory, query, extensions, top_k=2)

        # Then: Expect only files with .txt extension
        assert results == expected_results
        mock_sentence_transformer.encode.assert_called()
        mock_open.assert_called_with(
            "/test/dir/file1.txt", 'r', encoding='utf-8')

    def test_search_nonexistent_path(self, mock_sentence_transformer):
        """Test handling of nonexistent path."""
        # Given: A nonexistent path
        path = "/test/nonexistent"
        query = "test query"
        mock_os_path = patch(
            "jet.vectors.semantic_search.file_vector_search.os.path.exists", return_value=False)

        # When: Attempting to search a nonexistent path
        with mock_os_path:
            with pytest.raises(ValueError) as exc_info:
                search_files(path, query)

        # Then: Expect ValueError with correct message
        expected_error = f"Path {path} does not exist"
        assert str(exc_info.value) == expected_error

    def test_search_empty_directory(self, mock_sentence_transformer, mock_os_walk, mock_os_path):
        """Test searching an empty directory."""
        # Given: An empty directory
        directory = "/test/dir"
        query = "test query"
        expected_results = []
        mock_os_path.exists.return_value = True
        mock_os_path.isfile.return_value = False
        mock_os_path.isdir.return_value = True
        mock_os_walk.return_value = [("/test/dir", [], [])]

        # When: Searching the empty directory
        results = search_files(directory, query, top_k=2)

        # Then: Expect empty results
        assert results == expected_results
        mock_sentence_transformer.encode.assert_called_once()


class TestCollectFileChunks:
    def test_collect_file_chunks_single_file(self, mock_os_path, mock_open):
        """Test collecting chunks for a single file."""
        file_path = "/test/file1.txt"
        extensions = {".txt"}
        expected_paths = [file_path]
        expected_names = ["file1.txt"]
        expected_dirs = ["test"]
        expected_contents = [("sample content", 0, 13)]
        mock_os_path.exists.return_value = True
        mock_os_path.isfile.return_value = True
        mock_os_path.isdir.return_value = False
        mock_open.return_value.__enter__.return_value.read.return_value = "sample content"
        file_paths, file_names, parent_dirs, contents_with_indices = collect_file_chunks(
            file_path, extensions)  # Removed invalid 'reply' argument
        assert file_paths == expected_paths
        assert file_names == expected_names
        assert parent_dirs == expected_dirs
        assert contents_with_indices == expected_contents
        mock_open.assert_called_with(file_path, 'r', encoding='utf-8')

    def test_collect_file_chunks_directory(self, mock_os_walk, mock_os_path, mock_open):
        """Test collecting chunks for a directory."""
        # Given: A directory with two files
        directory = "/test/dir"
        extensions = {".txt"}
        expected_paths = ["/test/dir/file1.txt", "/test/dir/file2.txt"]
        expected_names = ["file1.txt", "file2.txt"]
        expected_dirs = ["dir", "dir"]
        expected_contents = [("sample content", 0, 13),
                             ("sample content", 0, 13)]
        mock_os_path.exists.return_value = True
        mock_os_path.isfile.return_value = False
        mock_os_path.isdir.return_value = True
        mock_os_walk.return_value = [
            ("/test/dir", [], ["file1.txt", "file2.txt"])]
        mock_open.return_value.__enter__.return_value.read.return_value = "sample content"

        # When: Collecting file chunks
        file_paths, file_names, parent_dirs, contents_with_indices = collect_file_chunks(
            directory, extensions)

        # Then: Expect correct file paths, names, directories, and content
        assert file_paths == expected_paths
        assert file_names == expected_names
        assert parent_dirs == expected_dirs
        assert contents_with_indices == expected_contents
        mock_os_walk.assert_called_once_with(directory)
        mock_open.assert_called()


class TestComputeWeightedSimilarity:
    """Tests for the compute_weighted_similarity function."""

    def test_compute_weighted_similarity_with_content(self):
        """Test computing weighted similarity with all components."""
        # Given: Query and component vectors
        query_vector = np.array([0.1, 0.2, 0.3])
        name_vector = np.array([0.4, 0.5, 0.6])
        dir_vector = np.array([0.3, 0.4, 0.5])
        content_vector = np.array([0.5, 0.6, 0.7])
        expected_result = (0.39, 0.4, 0.3, 0.5)  # 0.5*0.4 + 0.3*0.3 + 0.2*0.5

        # When: Computing weighted similarity
        result = compute_weighted_similarity(
            query_vector, name_vector, dir_vector, content_vector)

        # Then: Expect correct weighted and individual similarities
        assert pytest.approx(result[0], 0.01) == expected_result[0]
        assert pytest.approx(result[1], 0.01) == expected_result[1]
        assert pytest.approx(result[2], 0.01) == expected_result[2]
        assert pytest.approx(result[3], 0.01) == expected_result[3]

    def test_compute_weighted_similarity_no_content(self):
        """Test computing weighted similarity without content."""
        # Given: Query and component vectors, no content
        query_vector = np.array([0.1, 0.2, 0.3])
        name_vector = np.array([0.4, 0.5, 0.6])
        dir_vector = np.array([0.3, 0.4, 0.5])
        content_vector = None
        expected_result = (0.29, 0.4, 0.3, 0.0)  # 0.5*0.4 + 0.3*0.3 + 0.2*0.0

        # When: Computing weighted similarity
        result = compute_weighted_similarity(
            query_vector, name_vector, dir_vector, content_vector)

        # Then: Expect correct weighted and individual similarities
        assert pytest.approx(result[0], 0.01) == expected_result[0]
        assert pytest.approx(result[1], 0.01) == expected_result[1]
        assert pytest.approx(result[2], 0.01) == expected_result[2]
        assert pytest.approx(result[3], 0.01) == expected_result[3]


class TestGetFileVectors:
    """Tests for the get_file_vectors function."""

    def test_get_file_vectors_text_file(self, mock_sentence_transformer, mock_open, mock_pathlib_path):
        """Test get_file_vectors with a readable text file."""
        # Given: A text file with content
        file_path = "/test/file1.txt"
        mock_open.return_value.__enter__.return_value.read.return_value = "sample content"

        # When: Getting file vectors
        name_vector, dir_vector, content_vector = get_file_vectors(file_path)

        # Then: Expect correct vectors for file name, directory, and content
        assert np.array_equal(name_vector, np.array(
            [0.4, 0.5, 0.6]))  # file1.txt
        assert np.array_equal(dir_vector, np.array(
            [0.3, 0.4, 0.5]))    # test_dir
        assert np.array_equal(content_vector, np.array(
            [0.5, 0.6, 0.7]))  # sample content
        mock_open.assert_called_with(file_path, 'r', encoding='utf-8')
        mock_sentence_transformer.encode.assert_any_call(
            "file1.txt", convert_to_numpy=True)
        mock_sentence_transformer.encode.assert_any_call(
            "test", convert_to_numpy=True)
        mock_sentence_transformer.encode.assert_any_call(
            "sample content", convert_to_numpy=True)

    def test_get_file_vectors_non_text_file(self, mock_sentence_transformer, mock_open, mock_pathlib_path):
        """Test get_file_vectors with a non-text file."""
        # Given: A non-text file (e.g., .bin)
        file_path = "/test/file1.bin"
        mock_pathlib_path.return_value.suffix = ".bin"
        mock_open.side_effect = UnicodeDecodeError(
            "utf-8", b"", 0, 1, "invalid")

        # When: Getting file vectors
        name_vector, dir_vector, content_vector = get_file_vectors(file_path)

        # Then: Expect vectors for file name and directory, but no content vector
        assert np.array_equal(name_vector, np.array(
            [0.4, 0.5, 0.6]))  # file1.bin
        assert np.array_equal(dir_vector, np.array(
            [0.3, 0.4, 0.5]))    # test_dir
        assert content_vector is None
        mock_open.assert_called_with(file_path, 'r', encoding='utf-8')
        mock_sentence_transformer.encode.assert_any_call(
            "file1.bin", convert_to_numpy=True)
        mock_sentence_transformer.encode.assert_any_call(
            "test", convert_to_numpy=True)

    def test_get_file_vectors_root_directory(self, mock_sentence_transformer, mock_open, mock_pathlib_path):
        """Test get_file_vectors with a file in the root directory."""
        # Given: A file in the root directory
        file_path = "/file1.txt"
        mock_pathlib_path.return_value.parent.name = ""  # Simulate root directory
        mock_open.return_value.__enter__.return_value.read.return_value = "sample content"

        # When: Getting file vectors
        name_vector, dir_vector, content_vector = get_file_vectors(file_path)

        # Then: Expect correct vectors with root as parent directory
        assert np.array_equal(name_vector, np.array(
            [0.4, 0.5, 0.6]))  # file1.txt
        assert np.array_equal(dir_vector, np.array([0.3, 0.4, 0.5]))    # root
        assert np.array_equal(content_vector, np.array(
            [0.5, 0.6, 0.7]))  # sample content
        mock_open.assert_called_with(file_path, 'r', encoding='utf-8')
        mock_sentence_transformer.encode.assert_any_call(
            "file1.txt", convert_to_numpy=True)
        mock_sentence_transformer.encode.assert_any_call(
            "root", convert_to_numpy=True)
        mock_sentence_transformer.encode.assert_any_call(
            "sample content", convert_to_numpy=True)


class TestCosineSimilarity:
    """Tests for the cosine_similarity function."""

    def test_cosine_similarity_non_zero_vectors(self):
        """Test cosine similarity with non-zero vectors."""
        # Given: Two non-zero vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.5, 0.5, 0.0])
        expected_similarity = 0.7071067811865475  # cos(45°) ≈ 0.707

        # When: Calculating cosine similarity
        similarity = cosine_similarity(vec1, vec2)

        # Then: Expect correct similarity score
        assert pytest.approx(similarity, 0.01) == expected_similarity

    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with a zero vector."""
        # Given: One zero vector
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 0.0, 0.0])
        expected_similarity = 0.0

        # When: Calculating cosine similarity
        similarity = cosine_similarity(vec1, vec2)

        # Then: Expect zero similarity
        assert similarity == expected_similarity

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity with identical vectors."""
        # Given: Two identical vectors
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.0, 2.0, 3.0])
        expected_similarity = 1.0

        # When: Calculating cosine similarity
        similarity = cosine_similarity(vec1, vec2)

        # Then: Expect similarity of 1.0
        assert pytest.approx(similarity, 0.01) == expected_similarity
