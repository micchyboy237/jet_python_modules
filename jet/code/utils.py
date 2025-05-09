import os
from pathlib import Path
from jet.code.markdown_code_extractor import MarkdownCodeExtractor
from typing import List, TypedDict, Optional
import tokenize
import io
from tokenize import TokenInfo, STRING, COMMENT, NL, NEWLINE

from jet.file.utils import save_file


def remove_single_line_comments_preserving_triple_quotes(code):
    result = []
    tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))

    inside_string = False

    for i, tok in enumerate(tokens):
        token_type, token_string, start, end, line = tok

        if token_type == STRING:
            # Preserve the entire string token (including triple-quoted strings)
            result.append(tok)
            continue

        if token_type == COMMENT:
            # Remove the comment if not inside a string
            continue

        result.append(tok)

    # Rebuild the source code
    result_str = tokenize.untokenize(result)
    # Remove trailing spaces from each line
    result_str = '\n'.join(line.rstrip() for line in result_str.split('\n'))
    return result_str


class CodeBlock(TypedDict):
    language: str
    content: str


class MarkdownDoc(TypedDict):
    header: str
    header_level: int
    content: List[str]
    length: int


class ProcessedResult(TypedDict):
    header: str
    header_level: int
    length: int
    content: str
    code: Optional[CodeBlock]
    text: str


def process_markdown_file(md_file_path: str) -> list[ProcessedResult]:
    """
    Process a markdown file, extract code blocks, and save results as JSON.

    Args:
        md_file_path: Path to the input markdown file
        output_dir: Directory to save the output JSON file
    """
    from jet.code.splitter_markdown_utils import get_md_header_contents
    from jet.file.utils import load_file, save_file

    # Load markdown content
    md_text: str = load_file(md_file_path)

    # Get header contents
    docs: List[MarkdownDoc] = get_md_header_contents(md_text)

    # Initialize extractor
    extractor = MarkdownCodeExtractor()

    # Process documents
    results: List[ProcessedResult] = []
    for doc in docs:
        code_blocks: List[CodeBlock] = extractor.extract_code_blocks(
            doc["content"])

        # Prepare base result
        result: ProcessedResult = {
            "header": doc["header"],
            "header_level": doc["header_level"],
            "length": doc["length"],
            "content": "\n".join(doc["content"].splitlines()[1:]),
            "code": None,
            "text": doc["content"]
        }

        # Handle code blocks if present
        if code_blocks:
            result["content"] = extractor.remove_code_blocks(result["content"])
            result["code"] = {
                "language": code_blocks[0]["language"],
                "content": code_blocks[0]["code"]
            }
            result["length"] = len(result["text"])

        results.append(result)

    return results


def preprocess_notebooks_to_markdowns(md_file_or_dir: str, output_dir: str):
    def save_results(results: list[ProcessedResult], file: str, output_dir: str):
        # Prepare output file path
        file_name: str = f"{os.path.splitext(os.path.basename(file))[0]}.json"
        output_path: str = os.path.join(output_dir, file_name)

        # Save results
        save_file(results, output_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process the markdown file
    if os.path.isdir(md_file_or_dir):
        for file in Path(md_file_or_dir).glob('*.md'):
            results = process_markdown_file(str(file))
            save_results(results, str(file), output_dir)
    else:
        results = process_markdown_file(md_file_or_dir)
        save_results(results, md_file_or_dir, output_dir)
