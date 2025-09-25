import pytest
import os
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch
from jet.vectors.semantic_search.streaming_file_searcher import FileSearcher

@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory with test files."""
    logger.info(f"Creating temp directory at: {tmp_path}")
    node_modules_dir = tmp_path / "node_modules"
    node_modules_dir.mkdir(exist_ok=True)
    logger.info(f"Created node_modules directory at: {node_modules_dir}")

    (tmp_path / "doc1.txt").write_text("Machine learning is great")
    logger.info(f"Created file: {tmp_path / 'doc1.txt'}")
    (tmp_path / "doc2.py").write_text("import tensorflow")
    logger.info(f"Created file: {tmp_path / 'doc2.py'}")
    (node_modules_dir / "log.txt").write_text("Some log data")
    logger.info(f"Created file: {node_modules_dir / 'log.txt'}")
    (tmp_path / "irrelevant.doc").write_text("Unrelated content")
    logger.info(f"Created file: {tmp_path / 'irrelevant.doc'}")
    return tmp_path

class TestFileSearcher:
    def test_search_finds_relevant_files(self, temp_dir):
        # Given: A searcher with specific filters and a query
        searcher = FileSearcher(
            base_dir=str(temp_dir),
            threshold=0.5,
            includes=["*.txt", "*.py"],
            excludes=["*/node_modules/*"]
        )
        query = "machine learning"

        # When: Searching for relevant files
        with patch('jet.vectors.semantic_search.streaming_file_searcher.FileSearcher._get_embedding') as mock_embed:
            with patch('jet.vectors.semantic_search.streaming_file_searcher.FileSearcher._cosine_similarity') as mock_sim:
                mock_embed.side_effect = [
                    np.ones(768),  # query embedding
                    np.ones(768),  # doc1.txt
                    np.ones(768),  # doc2.py
                    np.zeros(768),  # irrelevant.doc
                    np.ones(768)   # node_modules/log.txt
                ]
                mock_sim.side_effect = [0.9, 0.9, 0.4, 0.4]

                results = sorted(list(searcher.search(query)), key=lambda x: x[0])

        # Then: Expect only relevant files with high scores
        expected = sorted([
            (str(temp_dir / "doc1.txt"), 0.9),
            (str(temp_dir / "doc2.py"), 0.9)
        ], key=lambda x: x[0])
        assert results == expected, "Should find only txt and py files with high relevance"

    def test_search_excludes_filtered_files(self, temp_dir):
        # Given: A searcher with exclude filter for node_modules
        searcher = FileSearcher(
            base_dir=str(temp_dir),
            threshold=0.5,
            excludes=["*/node_modules/*"]
        )
        query = "log data"

        # When: Searching with mock embeddings
        with patch('jet.vectors.semantic_search.streaming_file_searcher.FileSearcher._get_embedding') as mock_embed:
            with patch('jet.vectors.semantic_search.streaming_file_searcher.FileSearcher._cosine_similarity') as mock_sim:
                mock_embed.return_value = np.ones(768)
                mock_sim.return_value = 0.9

                results = sorted(list(searcher.search(query)), key=lambda x: x[0])

        # Then: Expect node_modules file to be excluded
        expected = sorted([
            (str(temp_dir / "doc1.txt"), 0.9),
            (str(temp_dir / "doc2.py"), 0.9),
            (str(temp_dir / "irrelevant.doc"), 0.9)
        ], key=lambda x: x[0])
        assert results == expected, "Should exclude node_modules files"

    def test_search_below_threshold(self, temp_dir):
        # Given: A searcher with high threshold
        searcher = FileSearcher(
            base_dir=str(temp_dir),
            threshold=0.8
        )
        query = "machine learning"

        # When: Searching with low similarity scores
        with patch('jet.vectors.semantic_search.streaming_file_searcher.FileSearcher._get_embedding') as mock_embed:
            with patch('jet.vectors.semantic_search.streaming_file_searcher.FileSearcher._cosine_similarity') as mock_sim:
                mock_embed.return_value = np.ones(768)
                mock_sim.return_value = 0.4

                results = list(searcher.search(query))

        # Then: Expect no results due to low scores
        expected = []
        assert results == expected, "Should return no files below threshold"

    def test_search_large_file_chunking(self, temp_dir):
        # Given: A searcher with small chunk size and a large file
        large_content = "Machine learning " * 1000  # ~17KB
        (temp_dir / "large.txt").write_text(large_content)
        searcher = FileSearcher(
            base_dir=str(temp_dir),
            threshold=0.5,
            includes=["*.txt"],
            chunk_size=512,
            chunk_overlap=50
        )
        query = "machine learning"

        # When: Searching with mock embeddings for chunks
        with patch('jet.vectors.semantic_search.streaming_file_searcher.FileSearcher._get_embedding') as mock_embed:
            with patch('jet.vectors.semantic_search.streaming_file_searcher.FileSearcher._cosine_similarity') as mock_sim:
                mock_embed.return_value = np.ones(768)
                # Simulate deterministic file order by sorting files
                files = sorted([f for f in temp_dir.rglob('*') if f.is_file() and searcher._is_included(f)], key=str)
                # Expect: doc1.txt (1 chunk), large.txt (~37 chunks), log.txt (1 chunk, excluded)
                score_map = {
                    str(temp_dir / "doc1.txt"): [0.9],
                    str(temp_dir / "large.txt"): [0.7, 0.8, 0.4] + [0.4] * 34,
                    str(temp_dir / "node_modules/log.txt"): [0.4]
                }
                # Flatten scores in order of sorted files
                mock_scores = []
                for file in files:
                    mock_scores.extend(score_map.get(str(file), [0.4]))
                mock_sim.side_effect = mock_scores

                results = sorted(list(searcher.search(query)), key=lambda x: x[0])

        # Then: Expect large file with highest chunk score
        expected = sorted([
            (str(temp_dir / "large.txt"), 0.9)  # Highest score from chunks
        ], key=lambda x: x[0])
        assert results == expected, "Should process large file in chunks and return highest score"
