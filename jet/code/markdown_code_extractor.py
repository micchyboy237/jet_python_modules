import re
from typing import List, Optional, TypedDict

CODE_BLOCK_PATTERN = r"```[ \t]*(\w+)?[ \t]*\n(.*?)\n```"
FILE_PATH_PATTERN = r".*File Path: `?([^`]+)`?"
UNKNOWN = "text"


class CodeBlock(TypedDict):
    """Represents a code block with associated metadata."""
    code: str  # The code content.
    language: str  # Programming language of the code block.
    file_path: Optional[str]  # The file path associated with the code block.
    # The file extension inferred from the language or file path.
    extension: Optional[str]


class MarkdownCodeExtractor:
    """Extracts code blocks from Markdown content."""

    LANGUAGE_EXTENSIONS = {
        "python": ".py",
        "javascript": ".js",
        "java": ".java",
        "html": ".html",
        "css": ".css",
        "cpp": ".cpp",
        "c": ".c",
        "ruby": ".rb",
        "go": ".go",
        "php": ".php",
        "cypher": ".cypher",
        "text": ".txt",  # Added for 'text' blocks.
        "unknown": "",
    }

    def get_extension(self, language: str, file_path: Optional[str]) -> str:
        """Determine the file extension based on the language or file path.

        Args:
            language (str): Programming language.
            file_path (Optional[str]): File path if available.

        Returns:
            str: File extension.
        """
        if file_path:
            return "." + file_path.split(".")[-1] if "." in file_path else ""
        return self.LANGUAGE_EXTENSIONS.get(language.lower(), "")

    def extract_code_blocks(self, markdown: str, with_text: bool = False) -> List[CodeBlock]:
        """Extract code blocks and associated file paths from Markdown text.

        Args:
            markdown (str): Markdown content.
            with_text (bool): Whether to include non-code text blocks as well.

        Returns:
            List[CodeBlock]: List of extracted code blocks.
        """
        lines = markdown.strip().splitlines()
        code_blocks: List[CodeBlock] = []
        current_file_path: Optional[str] = None
        inside_code_block = False
        lang = None
        code_lines = []
        text_content = ""

        for line in lines:
            # Check for file path pattern
            file_path_match = re.match(FILE_PATH_PATTERN, line)
            if file_path_match:
                current_file_path = file_path_match.group(1)
                continue

            # Handle start of a code block
            if line.startswith("```"):
                if not inside_code_block:
                    if text_content and with_text:
                        # Add the current text content as a text block if 'with_text' is True
                        code_blocks.append(
                            CodeBlock(
                                code=text_content,
                                language=UNKNOWN,
                                file_path=None,
                                extension=".txt"
                            )
                        )
                        text_content = ""  # Reset text content
                    inside_code_block = True
                    lang = line.strip("`").strip() or UNKNOWN
                    code_lines = []
                else:
                    # End of a code block
                    code_content = "\n".join(code_lines).rstrip()
                    if code_content:
                        extension = self.get_extension(lang, current_file_path)
                        code_blocks.append(
                            CodeBlock(
                                code=code_content,
                                language=lang,
                                file_path=current_file_path,
                                extension=extension
                            )
                        )
                    inside_code_block = False
                    current_file_path = None
                continue

            # Collect lines inside a code block
            if inside_code_block:
                code_lines.append(line)
            else:
                # Collect non-code block content as "text"
                text_content += line + "\n"

        # Add any remaining text as a "text" block if 'with_text' is True
        if text_content and with_text:
            code_blocks.append(
                CodeBlock(
                    code=text_content.strip(),
                    language="text",
                    file_path=None,
                    extension=".txt"
                )
            )

        return code_blocks

    def remove_code_blocks(self, markdown: str, keep_file_paths: bool = False) -> str:
        """Remove code blocks from Markdown content, optionally preserving file path lines.

        Args:
            markdown (str): Markdown content.
            keep_file_paths (bool): Whether to keep file path lines associated with code blocks.

        Returns:
            str: Markdown content with code blocks removed.
        """
        lines = markdown.strip().splitlines()
        result = []
        inside_code_block = False

        for line in lines:
            # Check for file path pattern
            file_path_match = re.match(FILE_PATH_PATTERN, line)
            if file_path_match:
                if keep_file_paths:
                    result.append(line)
                continue

            # Handle code block boundaries
            if line.startswith("```"):
                inside_code_block = not inside_code_block
                continue

            # Add non-code block lines to result
            if not inside_code_block:
                result.append(line)

        return "\n".join(result).rstrip()
