"""
Smart chunking that respects token boundaries and adds overlap.
Uses jet's production-grade chunking utilities.
"""

from typing import Dict, List, Tuple

from config import chunking, models
from jet.adapters.llama_cpp.chunking_utils import chunk_texts_with_data
from utils import logger


class CodeChunker:
    def __init__(self):
        self.chunk_size = chunking.CHUNK_SIZE
        self.overlap = chunking.CHUNK_OVERLAP
        self.model = models.EMBED_MODEL  # Use embedding model's tokenizer

    def chunk_files(self, files: List[Tuple[str, str, str]]) -> List[Dict]:
        """
        Input: List of (path, lang, content)
        Output: List of chunk dicts ready for embedding

        Uses token-aware chunking with proper overlap handling.
        """
        logger.info(f"Chunking {len(files)} files...")

        all_chunks = []
        for path, lang, content in files:
            if not content.strip():
                continue

            # Use jet's chunking utility with metadata
            chunk_results = chunk_texts_with_data(
                texts=content,
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                model=self.model,
                ids=[f"{path}"],  # Document ID
                buffer=0,
                strict_sentences=False,  # Code doesn't have sentences
                min_chunk_size=32,
                show_progress=False,
            )

            # Convert to repo-agent format
            for chunk_result in chunk_results:
                # Calculate line numbers from character offsets
                start_line = content[: chunk_result["start_idx"]].count("\n") + 1
                end_line = content[: chunk_result["end_idx"]].count("\n") + 1

                chunk_dict = {
                    "content": chunk_result["content"],
                    "file_path": path,
                    "language": lang,
                    "start_line": start_line,
                    "end_line": end_line,
                    "id": f"{path}:{start_line}-{end_line}",
                    "num_tokens": chunk_result["num_tokens"],
                }
                all_chunks.append(chunk_dict)

        logger.info(f"Generated {len(all_chunks)} total chunks.")
        return all_chunks
