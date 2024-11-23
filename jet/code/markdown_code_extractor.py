import re
from typing import List, Optional, TypedDict

CODE_BLOCK_PATTERN = r"```[ \t]*(\w+)?[ \t]*\n(.*?)\n```"
FILE_PATH_PATTERN = r".*File Path: `?([^`]+)`?"
UNKNOWN = "unknown"


class CodeBlock(TypedDict):
    """Represents a code block with associated metadata."""
    code: str  # The code content.
    language: str  # Programming language of the code block.
    file_path: Optional[str]  # The file path associated with the code block.


class MarkdownCodeExtractor:
    """Extracts code blocks from Markdown content."""

    def extract_code_blocks(self, markdown: str) -> List[CodeBlock]:
        """Extract code blocks and associated file paths from Markdown text.

        Args:
            markdown (str): Markdown content.

        Returns:
            List[CodeBlock]: List of extracted code blocks.
        """
        lines = markdown.strip().splitlines()
        code_blocks: List[CodeBlock] = []
        current_file_path: Optional[str] = None
        inside_code_block = False
        lang = None
        code_lines = []

        for line in lines:
            line = line.strip()
            # Check for file path pattern
            file_path_match = re.match(FILE_PATH_PATTERN, line)
            if file_path_match:
                current_file_path = file_path_match.group(1)
                continue

            # Handle start of a code block
            if line.startswith("```"):
                if not inside_code_block:
                    inside_code_block = True
                    lang = line.strip("`").strip() or UNKNOWN
                    code_lines = []
                else:
                    # End of a code block
                    code_content = "\n".join(code_lines).strip()
                    if code_content:
                        code_blocks.append(
                            CodeBlock(
                                code=code_content,
                                language=lang,
                                file_path=current_file_path
                            )
                        )
                    inside_code_block = False
                    current_file_path = None
                continue

            # Collect lines inside a code block
            if inside_code_block:
                code_lines.append(line)

        return code_blocks
