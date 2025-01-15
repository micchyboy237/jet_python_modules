from pathlib import Path
import re
from llama_index.readers.file import MarkdownReader as BaseMarkdownReader
from typing import Optional, List, Tuple
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem


class MarkdownReader(BaseMarkdownReader):
    """Custom Markdown Reader with an option to preserve headers."""

    def __init__(
        self,
        *args,
        preserve_headers: bool = False,  # New option to prevent removing headers
        **kwargs
    ) -> None:
        """Initialize the custom markdown reader."""
        super().__init__(*args, **kwargs)
        self._preserve_headers = preserve_headers

    def markdown_to_tups(self, markdown_text: str) -> List[Tuple[Optional[str], str]]:
        """Convert a markdown file to tuples with an option to preserve headers."""
        markdown_tups: List[Tuple[Optional[str], str]] = []
        lines = markdown_text.split("\n")

        current_header = None
        current_lines = []
        in_code_block = False

        for line in lines:
            if line.startswith("```"):
                # Toggle code block state
                in_code_block = not in_code_block

            header_match = re.match(r"^#+\s", line)
            if not in_code_block and header_match:
                # Process headers
                if current_header is not None or len(current_lines) > 0:
                    markdown_tups.append(
                        (current_header, "\n".join(current_lines)))

                current_header = line if self._preserve_headers else re.sub(
                    r"#", "", line).strip()
                current_lines.clear()
            else:
                current_lines.append(line)

        # Append the final chunk
        markdown_tups.append((current_header, "\n".join(current_lines)))

        # Apply further processing for hyperlinks and images if enabled
        return [
            (
                key,
                re.sub(r"<.*?>", "",
                       self.remove_hyperlinks(self.remove_images(value)))
                if self._remove_hyperlinks or self._remove_images
                else value,
            )
            for key, value in markdown_tups
        ]

    def parse_tups(
        self,
        filepath: Path,
        errors: str = "ignore",
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Tuple[Optional[str], str]]:
        """Parse file into tuples with the preserve_headers option applied."""
        fs = fs or LocalFileSystem()
        with fs.open(filepath, encoding="utf-8") as f:
            content = f.read().decode(encoding="utf-8")
        if self._remove_hyperlinks:
            content = self.remove_hyperlinks(content)
        if self._remove_images:
            content = self.remove_images(content)
        return self.markdown_to_tups(content)


# Example Usage
if __name__ == "__main__":
    input_dir = "/path/to/markdown/files"
    reader = MarkdownReader(
        preserve_headers=True,  # Enable the option to preserve headers
        remove_hyperlinks=False,
        remove_images=False,
    )

    from llama_index.core.readers.file.base import SimpleDirectoryReader

    documents = SimpleDirectoryReader(
        input_dir,
        required_exts=[".md"],
        file_extractor={".md": reader},
    ).load_data()

    for doc in documents:
        print(doc.text)
