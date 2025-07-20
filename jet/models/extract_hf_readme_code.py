import re
from typing import List, Optional, TypedDict
from pathlib import Path
from jet.logger import logger
from jet.models.model_types import ModelType
from jet.models.utils import resolve_model_key

# MarkdownCodeExtractor and related definitions
CODE_BLOCK_PATTERN = r"```[ \t]*(\w+)?[ \t]*\n(.*?)\n```"
FILE_PATH_PATTERN = r".*File Path: `?([^`]+)`?"
UNKNOWN = "text"


class CodeBlock(TypedDict):
    """Represents a code block with associated metadata."""
    code: str
    language: str
    file_path: Optional[str]
    extension: str
    preceding_text: str
    index: int


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
        "text": ".txt",
        "unknown": "",
    }

    def get_extension(self, language: str, file_path: Optional[str]) -> str:
        if file_path:
            return "." + file_path.split(".")[-1] if "." in file_path else ""
        return self.LANGUAGE_EXTENSIONS.get(language.lower(), "")

    def extract_code_blocks(self, markdown: str, with_text: bool = False) -> List[CodeBlock]:
        lines = markdown.strip().splitlines()
        code_blocks: List[CodeBlock] = []
        current_file_path: Optional[str] = None
        inside_code_block = False
        lang = None
        code_lines = []
        text_content = ""
        current_index = 0
        text_start_index = 0

        for line in lines:
            file_path_match = re.match(FILE_PATH_PATTERN, line)
            if file_path_match:
                current_file_path = file_path_match.group(1)
                if text_content and with_text:
                    code_blocks.append(
                        CodeBlock(
                            code=text_content.rstrip(),
                            language=UNKNOWN,
                            file_path=None,
                            extension=".txt",
                            preceding_text="",
                            index=text_start_index
                        )
                    )
                    text_content = ""
                continue
            if line.startswith("```"):
                if not inside_code_block:
                    if text_content and with_text:
                        code_blocks.append(
                            CodeBlock(
                                code=text_content.rstrip(),
                                language=UNKNOWN,
                                file_path=None,
                                extension=".txt",
                                preceding_text="",
                                index=text_start_index
                            )
                        )
                        text_content = ""
                    inside_code_block = True
                    lang = line.strip("`").strip() or UNKNOWN
                    code_lines = []
                    text_start_index = current_index
                else:
                    code_content = "\n".join(code_lines).rstrip()
                    if code_content:
                        extension = self.get_extension(lang, current_file_path)
                        code_blocks.append(
                            CodeBlock(
                                code=code_content,
                                language=lang,
                                file_path=current_file_path,
                                extension=extension,
                                preceding_text=text_content.rstrip() if with_text else "",
                                index=current_index
                            )
                        )
                    inside_code_block = False
                    current_file_path = None
                    text_content = ""
                    text_start_index = current_index + 1
                continue
            if inside_code_block:
                code_lines.append(line)
            else:
                text_content += line + "\n"
            current_index += 1

        if text_content and with_text:
            code_blocks.append(
                CodeBlock(
                    code=text_content.rstrip(),
                    language="text",
                    file_path=None,
                    extension=".txt",
                    preceding_text="",
                    index=text_start_index
                )
            )
        return code_blocks


def extract_code_from_hf_readmes(
    input_dir: str, output_dir: str, model_id: ModelType, include_text: bool = True
):
    """Process the .md file for the specified model_id and extract code blocks."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    extractor = MarkdownCodeExtractor()

    # Construct the expected README filename for the model
    model_key = resolve_model_key(model_id)
    md_file = input_path / f"{model_key}_README.md"

    if not md_file.exists():
        logger.error(f"README file not found for {model_id}: {md_file}")
        return

    model_name = md_file.stem.replace("_README", "")
    sanitized_model_name = model_name.replace(":", "_")
    print(f"Processing {md_file.name}...")

    with open(md_file, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    code_blocks = extractor.extract_code_blocks(
        markdown_content, with_text=include_text
    )

    model_dir = output_path / sanitized_model_name
    model_dir.mkdir(exist_ok=True)
    lang_counters = {}

    for block in code_blocks:
        lang = block["language"].lower()
        ext = block["extension"] or ".txt"
        code = block["code"]
        file_path = block["file_path"]
        preceding_text = block["preceding_text"]
        lang_counters[lang] = lang_counters.get(lang, 0) + 1
        counter = lang_counters[lang]
        lang_dir = model_dir / lang
        lang_dir.mkdir(exist_ok=True)

        if file_path:
            filename = Path(file_path).name
        else:
            filename = f"{sanitized_model_name}_{counter}{ext}"

        filename = "".join(
            c for c in filename if c.isalnum() or c in (".", "_", "-")
        )
        output_file = lang_dir / filename

        with open(output_file, "w", encoding="utf-8") as f:
            if preceding_text and lang != "text":
                f.write(f"/* {preceding_text} */\n")
            f.write(code)

        logger.success(f"  Saved ({lang}): {output_file}")

    if not code_blocks:
        logger.debug(f"  No code blocks found in {md_file.name}")
