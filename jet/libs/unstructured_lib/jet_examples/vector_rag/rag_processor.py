"""DocumentProcessor: small, focused class for loading + smart chunking with unstructured."""

from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from rich.console import Console
from tqdm import tqdm
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition
from unstructured.partition.common import UnsupportedFileFormatError

from .rag_document import ChunkList

console = Console()

PartitionStrategyLiteral = Literal["auto", "fast", "ocr_only", "hi_res"]


class DocumentProcessor:
    """Generic processor - configurable strategy/chunk size, no hard-coded paths or logic."""

    def __init__(
        self,
        max_characters: int = 1000,
        new_after_n_chars: int = 500,
        combine_text_under_n_chars: int = 200,
        allowed_extensions: Optional[Iterable[str]] = None,
        strategy: PartitionStrategyLiteral = "fast",  # "fast" = lightweight cross-platform (M1/Win); override to "hi_res" after deps
    ):
        self.max_characters = max_characters
        self.new_after_n_chars = new_after_n_chars
        # Defensive clamp – satisfies unstructured.chunk_by_title requirement
        self.combine_text_under_n_chars = min(
            combine_text_under_n_chars, max_characters
        )
        self.strategy = strategy

        # Normalize extensions to lowercase with leading dot
        self.allowed_extensions = (
            {
                ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                for ext in allowed_extensions
            }
            if allowed_extensions
            else None
        )

        # temporary debug (remove after all tests pass)
        console.print(
            f"[yellow]DEBUG DocumentProcessor__init__: max={max_characters} combine={self.combine_text_under_n_chars}[/yellow]"
        )

    def _partition(self, file_path: str, **kwargs: Any) -> list:
        """Tiny private method: partition single file."""
        return partition(filename=file_path, strategy=self.strategy, **kwargs)

    def _chunk(self, elements: list) -> list:
        """Tiny private method: structure-aware chunking (preserves titles/sections for RAG)."""
        return chunk_by_title(
            elements,
            max_characters=self.max_characters,
            new_after_n_chars=self.new_after_n_chars,
            combine_text_under_n_chars=self.combine_text_under_n_chars,
        )

    def process_file(self, file_path: str, **kwargs: Any) -> ChunkList:
        """Public API: partition + chunk + convert to reusable Chunk."""

        path = Path(file_path)

        # Optional extension filtering
        if self.allowed_extensions is not None:
            if path.suffix.lower() not in self.allowed_extensions:
                return []

        try:
            elements = self._partition(file_path, **kwargs)
        except UnsupportedFileFormatError:
            console.print(
                f"[red]Skipping unsupported file:[/red] {Path(file_path).name}"
            )
            return []
        except Exception as e:
            console.print(
                f"[red]Error processing file {Path(file_path).name}: {e}[/red]"
            )
            return []

        chunked_elements = self._chunk(elements)
        console.print(
            f"[yellow]DEBUG process_file: produced {len(chunked_elements)} chunks from {Path(file_path).name}[/yellow]"
        )
        chunks: ChunkList = []
        for el in chunked_elements:
            text = getattr(el, "text", str(el)).strip()
            if text:
                metadata = (
                    getattr(el.metadata, "to_dict", lambda: dict(el.metadata))()
                    if el.metadata
                    else {}
                )
                chunks.append({"text": text, "metadata": metadata})
        return chunks

    def process_directory(self, directory: str) -> ChunkList:
        """Reusable batch processor with progress (tqdm + rich)."""
        all_chunks: ChunkList = []
        paths = list(Path(directory).rglob("*"))

        for path in tqdm(paths, desc="Processing files"):
            if path.is_file():
                chunks = self.process_file(str(path))
                all_chunks.extend(chunks)
        console.print(
            f"[green]Processed {len(all_chunks)} chunks from {directory}[/green]"
        )
        return all_chunks
