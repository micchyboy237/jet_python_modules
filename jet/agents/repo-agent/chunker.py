"""
Smart chunking that respects line boundaries and adds overlap.
"""

from typing import Dict, List

from config import chunking
from utils import logger


class CodeChunker:
    def __init__(self):
        self.chunk_size = chunking.CHUNK_SIZE
        self.overlap = chunking.CHUNK_OVERLAP

    def chunk_files(self, files: List[tuple]) -> List[Dict]:
        """
        Input: List of (path, lang, content)
        Output: List of chunk dicts ready for embedding
        """
        all_chunks = []
        for path, lang, content in files:
            chunks = self._split_content(content, path, lang)
            all_chunks.extend(chunks)

        logger.info(f"Generated {len(all_chunks)} total chunks.")
        return all_chunks

    def _split_content(self, content: str, path: str, lang: str) -> List[Dict]:
        lines = content.split("\n")
        chunks = []
        current_chunk_lines = []
        current_token_count = 0
        start_line = 1

        for i, line in enumerate(lines):
            # More accurate token estimation for code: ~4 chars per token
            line_tokens = max(1, len(line) // 4)
            new_total = current_token_count + line_tokens

            if new_total > self.chunk_size and current_chunk_lines:
                # Save current chunk
                chunk_text = "\n".join(current_chunk_lines)
                chunks.append(
                    {
                        "content": chunk_text,
                        "file_path": path,
                        "language": lang,
                        "start_line": start_line,
                        "end_line": i,
                        "id": f"{path}:{start_line}-{i}",
                    }
                )

                # Handle overlap
                overlap_lines = []
                overlap_tokens = 0
                # Go backwards from the current chunk to find overlap
                for prev_line in reversed(current_chunk_lines):
                    lt = max(1, len(prev_line) // 4)
                    if overlap_tokens + lt > self.overlap:
                        break
                    overlap_lines.insert(0, prev_line)
                    overlap_tokens += lt

                current_chunk_lines = overlap_lines
                start_line = i - len(overlap_lines) + 1
                current_token_count = overlap_tokens
            else:
                current_token_count = new_total

            current_chunk_lines.append(line)

        # Don't forget the last chunk
        if current_chunk_lines:
            chunks.append(
                {
                    "content": "\n".join(current_chunk_lines),
                    "file_path": path,
                    "language": lang,
                    "start_line": start_line,
                    "end_line": len(lines),
                    "id": f"{path}:{start_line}-{len(lines)}",
                }
            )

        return chunks
