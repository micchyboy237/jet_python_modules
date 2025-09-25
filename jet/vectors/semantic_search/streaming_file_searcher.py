import os
import fnmatch
from typing import List, Iterator, Tuple, Optional
from pathlib import Path
import numpy as np
from ollama import Client
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileSearcher:
    def __init__(
        self,
        base_dir: str,
        threshold: float = 0.5,
        includes: Optional[List[str]] = None,
        excludes: Optional[List[str]] = None,
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "nomic-embed-text",
        large_file_size: int = 1024 * 1024,  # 1MB
        chunk_size: int = 512,  # Bytes to read per chunk
        chunk_overlap: int = 50,  # Bytes to overlap between chunks
    ):
        self.base_dir = Path(base_dir).resolve()
        self.threshold = threshold
        self.includes = includes or []
        self.excludes = excludes or []
        self.large_file_size = large_file_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ollama_client = Client(host=ollama_host)
        self.ollama_model = ollama_model

    def _matches_patterns(self, path: Path, patterns: List[str]) -> bool:
        """Check if path matches any of the given patterns."""
        return any(fnmatch.fnmatch(str(path), pattern) or fnmatch.fnmatch(path.name, pattern) for pattern in patterns)

    def _is_included(self, path: Path) -> bool:
        """Check if file should be included based on include/exclude patterns."""
        if self.excludes and self._matches_patterns(path, self.excludes):
            return False
        if not self.includes or self._matches_patterns(path, self.includes):
            return True
        return False

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for given text using Ollama."""
        try:
            response = self.ollama_client.embeddings(model=self.ollama_model, prompt=text)
            return np.array(response['embedding'])
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(768)  # Fallback embedding size for nomic-embed-text

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot_product / norm_product if norm_product != 0 else 0.0

    def _read_file_in_chunks(self, file_path: Path) -> Iterator[str]:
        """Read file in chunks with overlap."""
        logger.debug(f"Reading file in chunks: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_size = file_path.stat().st_size
                logger.debug(f"File size: {file_size}, chunk_size: {self.chunk_size}, overlap: {self.chunk_overlap}")
                if file_size <= self.chunk_size:
                    chunk = f.read(self.chunk_size)
                    logger.debug(f"Read single chunk of size {len(chunk)} for {file_path}")
                    yield chunk
                    return

                pos = 0
                chunk_count = 0
                while pos < file_size:
                    f.seek(pos)
                    chunk = f.read(self.chunk_size)
                    chunk_count += 1
                    logger.debug(f"Read chunk {chunk_count} at pos {pos}, size {len(chunk)} for {file_path}")
                    if not chunk:
                        break
                    yield chunk
                    pos += self.chunk_size - self.chunk_overlap
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
            yield ""

    def search(self, query: str) -> Iterator[Tuple[str, float]]:
        """Stream files with relevance scores above threshold."""
        logger.debug(f"Starting search with query: {query}")
        query_embedding = self._get_embedding(query)
        total_files = sum(1 for _ in self.base_dir.rglob('*') if _.is_file() and self._is_included(_))
        total_folders = sum(1 for _ in self.base_dir.rglob('*') if _.is_dir())

        current_folder = None
        folder_count = 0

        with tqdm(total=total_files, desc="Searching files", unit="file") as file_pbar:
            for file_path in self.base_dir.rglob('*'):
                if file_path.is_dir():
                    if file_path != current_folder:
                        current_folder = file_path
                        folder_count += 1
                        logger.info(f"Processing folder {folder_count}/{total_folders}: {file_path}")
                    continue

                logger.debug(f"Checking file: {file_path}")
                if not self._is_included(file_path):
                    logger.debug(f"File excluded by filters: {file_path}")
                    file_pbar.update(1)
                    continue

                try:
                    file_size = file_path.stat().st_size
                    if file_size > self.large_file_size:
                        logger.info(f"Processing large file ({file_size / 1024 / 1024:.2f} MB): {file_path}")

                    max_score = 0.0
                    chunk_count = 0
                    for chunk in self._read_file_in_chunks(file_path):
                        chunk_count += 1
                        logger.debug(f"Processing chunk {chunk_count} for {file_path}")
                        if not chunk:
                            logger.debug(f"Empty chunk for {file_path}")
                            continue
                        file_embedding = self._get_embedding(chunk)
                        score = self._cosine_similarity(query_embedding, file_embedding)
                        logger.debug(f"Chunk {chunk_count} score: {score} for {file_path}")
                        max_score = max(max_score, score)

                    logger.debug(f"Max score for {file_path}: {max_score}")
                    if max_score > self.threshold:
                        logger.info(f"Found relevant file: {file_path} (score: {max_score:.3f})")
                        yield str(file_path), max_score

                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
                finally:
                    file_pbar.update(1)

if __name__ == "__main__":
    searcher = FileSearcher(
        base_dir=".",
        threshold=0.5,
        includes=["*.txt", "*.py"],
        excludes=["*/node_modules/*", "*.log"],
        ollama_host="http://localhost:11434",
        ollama_model="nomic-embed-text",
        large_file_size=1024 * 1024,
        chunk_size=512,
        chunk_overlap=50
    )
    query = "machine learning"
    for file_path, score in searcher.search(query):
        print(f"File: {file_path}, Relevance Score: {score:.3f}")