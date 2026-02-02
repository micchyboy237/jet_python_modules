import re
from typing import TypedDict

CODE_BLOCK_PATTERN = r"```[ \t]*(\w+)?[ \t]*\n(.*?)\n```"
FILE_PATH_PATTERN = r".*File Path: `?([^`]+)`?"
UNKNOWN = "text"


class CodeBlock(TypedDict):
    """Represents a code block with associated metadata."""

    code: str  # The code content.
    language: str  # Programming language of the code block.
    file_path: str | None  # The file path associated with the code block.
    # The file extension inferred from the language or file path.
    extension: str | None


class MarkdownCodeExtractor:
    """Extracts code blocks from Markdown content."""

    LANGUAGE_ALIASES = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "jsx": "javascript",  # or "jsx"
        "rb": "ruby",
        "yml": "yaml",
        "yaml": "yaml",
        "md": "markdown",
        "txt": "text",
        "text": "text",
        "sh": "bash",
        "bash": "bash",
        "zsh": "bash",
        "shell": "bash",
        "json": "json",
        "html": "html",
        "css": "css",
        # add more as needed
    }

    LANGUAGE_EXTENSIONS = {
        "python": ".py",
        "javascript": ".js",
        "typescript": ".ts",
        "ruby": ".rb",
        "yaml": ".yaml",  # or .yml—choose one
        "text": ".txt",
        "markdown": ".md",
        "bash": ".sh",
        "json": ".json",
        "html": ".html",
        "css": ".css",
        "java": ".java",
        "cpp": ".cpp",
        "c": ".c",
        "go": ".go",
        "php": ".php",
        "cypher": ".cypher",
        "unknown": "",
    }

    def normalize_language(self, lang: str) -> str:
        """Convert short alias or any case variation → canonical lowercase name"""
        if not lang:
            return "text"  # or "unknown"
        lang = lang.strip().lower()
        return self.LANGUAGE_ALIASES.get(lang, lang)

    def get_extension(self, language: str, file_path: str | None = None) -> str:
        """Determine file extension based on normalized language or file path"""
        if file_path:
            ext = file_path.rsplit(".", 1)[-1].lower()
            if ext and "." in file_path:
                return f".{ext}"
        normalized = self.normalize_language(language)
        return self.LANGUAGE_EXTENSIONS.get(normalized, "")

    def extract_code_blocks(
        self, markdown: str, with_text: bool = False
    ) -> list[CodeBlock]:
        """Extract code blocks and associated file paths from Markdown text.

        Args:
            markdown (str): Markdown content.
            with_text (bool): Whether to include non-code text blocks as well.

        Returns:
            List[CodeBlock]: List of extracted code blocks.
        """
        lines = markdown.strip().splitlines()
        code_blocks: list[CodeBlock] = []
        current_file_path: str | None = None
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

            # Handle start or end of a code block
            if line.startswith("```"):
                if not inside_code_block:
                    if text_content and with_text:
                        # Add the current text content as a text block if 'with_text' is True
                        code_blocks.append(
                            CodeBlock(
                                code=text_content,
                                language=UNKNOWN,
                                file_path=None,
                                extension=".txt",
                            )
                        )
                        text_content = ""  # Reset text content
                    inside_code_block = True
                    lang = self.normalize_language(line.strip("`").strip())
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
                                extension=extension,
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
                    extension=".txt",
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
