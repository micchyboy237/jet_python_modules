import json
import os
import shutil
import sys
from typing import List, Literal, Union
from pathlib import Path
from jet.file.utils import save_file
from jet.code.utils import remove_single_line_comments_preserving_triple_quotes
from jet.code.markdown_code_extractor import MarkdownCodeExtractor, CodeBlock
from jet.logger import logger


def extract_text_from_ipynb(
    notebook_path,
    include_outputs=True,
    include_code=False,
    include_comments=False,
    merge_consecutive_code=False,
    save_as: Literal['md', 'py'] = 'md'
):
    """Extract text content from a Jupyter notebook and return as markdown or Python code."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        content = []
        text_blocks = []
        last_code_buffer = []

        def flush_code_buffer():
            """Helper to flush buffered consecutive code cells."""
            nonlocal last_code_buffer
            if not last_code_buffer:
                return
            code_content = "\n".join(last_code_buffer)
            content.append("```python")
            content.append(code_content)
            content.append("```")
            content.append("")
            last_code_buffer = []

        for cell in notebook.get('cells', []):
            if cell['cell_type'] == 'markdown':
                # flush any buffered code first
                if merge_consecutive_code:
                    flush_code_buffer()

                if save_as == 'md':
                    content.extend(cell['source'])
                    content.append("")
                else:
                    text_blocks.extend(cell['source'])
                    text_blocks.append("")

            elif cell['cell_type'] == 'code':
                if text_blocks and save_as == 'py':
                    content.append('"""')
                    content.extend(line.rstrip()
                                   for line in text_blocks if line.strip())
                    content.append('"""')
                    content.append('')
                    text_blocks = []

                if include_code:
                    code_content = ''.join(cell['source'])
                    if not include_comments:
                        code_content = remove_single_line_comments_preserving_triple_quotes(
                            code_content)

                    if save_as == 'md':
                        if merge_consecutive_code:
                            last_code_buffer.append(code_content)
                        else:
                            content.append("```python")
                            content.append(code_content)
                            content.append("```")
                            content.append("")
                    else:  # save_as == 'py'
                        content.append(code_content)
                        content.append("")

                if include_outputs and save_as == 'md':
                    if merge_consecutive_code:
                        flush_code_buffer()
                    for output in cell.get('outputs', []):
                        if 'text' in output:
                            content.append("```output")
                            content.extend(output['text'])
                            content.append("```")
                            content.append("")
                        elif 'data' in output and 'text/plain' in output['data']:
                            content.append("```output")
                            content.extend(output['data']['text/plain'])
                            content.append("```")
                            content.append("")

        if text_blocks and save_as == 'py':
            content.append('"""')
            content.extend(line.rstrip()
                           for line in text_blocks if line.strip())
            content.append('"""')
            content.append('')

        if merge_consecutive_code:
            flush_code_buffer()

        return "\n".join(line.rstrip() for line in content)

    except Exception as e:
        print(f"Error processing {notebook_path}: {str(e)}")
        return None


def extract_text_from_md(
    md_path,
    include_code=True,
    include_comments=True,
    merge_consecutive_code=False,
    save_as: Literal['md', 'py'] = 'md'
):
    """Extract text content from a markdown file using MarkdownCodeExtractor."""
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        extractor = MarkdownCodeExtractor()
        if save_as == 'md':
            if include_code:
                code_blocks = extractor.extract_code_blocks(
                    content, with_text=True)
                markdown_content = []
                buffer_lang = None
                buffer_code = []

                def flush_buffer():
                    nonlocal buffer_lang, buffer_code
                    if buffer_lang and buffer_code:
                        markdown_content.append(f"```{buffer_lang}")
                        markdown_content.append("\n".join(buffer_code))
                        markdown_content.append("```")
                        markdown_content.append("")
                    buffer_lang, buffer_code = None, []

                for block in code_blocks:
                    if block['language'] == 'text':
                        if merge_consecutive_code:
                            flush_buffer()
                        markdown_content.append(block['code'])
                        markdown_content.append("")
                    else:
                        code_content = block['code']
                        if not include_comments and block['language'] == 'python':
                            code_content = remove_single_line_comments_preserving_triple_quotes(
                                code_content)

                        if merge_consecutive_code:
                            if buffer_lang == block['language']:
                                buffer_code.append(code_content)
                            else:
                                flush_buffer()
                                buffer_lang = block['language']
                                buffer_code = [code_content]
                        else:
                            markdown_content.append(f"```{block['language']}")
                            markdown_content.append(code_content)
                            markdown_content.append("```")
                            markdown_content.append("")

                if merge_consecutive_code:
                    flush_buffer()

                return "\n".join(line.rstrip() for line in markdown_content)
            else:
                return extractor.remove_code_blocks(content, keep_file_paths=False)

        # (py mode left unchanged for brevity but could apply same idea)
        ...
    except Exception as e:
        print(f"Error processing {md_path}: {str(e)}")
        return None


def extract_text_from_py(py_path, include_code=True, include_comments=True, save_as: Literal['md', 'py'] = 'md'):
    """Extract text content from a Python file as markdown or Python code."""
    # Implement using ast
    pass


def process_file(
    input_path,
    output_dir=None,
    include_outputs=True,
    include_code=False,
    include_comments=False,
    merge_consecutive_code=False,
    save_as: Literal['md', 'py'] = 'md'
):
    """Process a single file (notebook, markdown, or python) and save as markdown or Python code."""
    input_path = Path(input_path)

    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{input_path.stem}.{save_as}"

    if input_path.suffix == '.ipynb':
        content = extract_text_from_ipynb(
            input_path,
            include_outputs=include_outputs,
            include_code=include_code,
            include_comments=include_comments,
            merge_consecutive_code=merge_consecutive_code,
            save_as=save_as
        )
    elif input_path.suffix == '.md':
        content = extract_text_from_md(
            input_path,
            include_code=include_code,
            include_comments=include_comments,
            merge_consecutive_code=merge_consecutive_code,
            save_as=save_as
        )
    elif input_path.suffix == '.py':
        content = extract_text_from_py(
            input_path,
            include_code=include_code,
            include_comments=include_comments,
            merge_consecutive_code=merge_consecutive_code,
            save_as=save_as
        )
    else:
        print(f"Unsupported file type: {input_path}")
        return

    if content:
        save_file(content, str(output_path))


def run_text_extraction(
    input_path: str,
    output_dir: str,
    extensions: Union[str, List[str]] = ['.ipynb', '.md', '.py'],
    save_as: Literal['md', 'py'] = 'md',
    include_outputs: bool = False,
    include_code: bool = False,
    include_comments: bool = True,
    merge_consecutive_code: bool = False,
):
    """Main function to process notebook, markdown, and python files."""
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    extensions = [extensions] if isinstance(extensions, str) else extensions

    if os.path.isdir(input_path):
        for ext in extensions:
            for file in Path(input_path).rglob(f"*.{ext.lstrip('*.')}"):
                process_file(
                    file,
                    output_dir,
                    include_outputs=include_outputs,
                    include_code=include_code,
                    include_comments=include_comments,
                    merge_consecutive_code=merge_consecutive_code,
                    save_as=save_as
                )
    else:
        process_file(
            input_path,
            output_dir,
            include_outputs=include_outputs,
            include_code=include_code,
            include_comments=include_comments,
            merge_consecutive_code=merge_consecutive_code,
            save_as=save_as
        )


def run_notebook_extraction(
    input_path: str,
    output_dir: str,
    save_as: Literal['md', 'py'] = 'md',
    include_outputs: bool = False,
    include_code: bool = False,
    include_comments: bool = True,
    merge_consecutive_code: bool = False,
):
    """Main function to process notebook files."""
    extension = "*.ipynb"
    run_text_extraction(
        input_path,
        output_dir,
        extension,
        save_as,
        include_outputs=include_outputs,
        include_code=include_code,
        include_comments=include_comments,
        merge_consecutive_code=merge_consecutive_code,
    )


if __name__ == "__main__":
    input_path = "/Users/jethroestrada/Desktop/External_Projects/AI/rag_05_2025/RAG_Techniques/all_rag_techniques"
    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/RAG_Techniques/all_rag_techniques"

    logger.info("Extracting texts from notebooks...")
    run_notebook_extraction(input_path, output_dir)

    # logger.info("Extracting documentation markdown...")
    # run_text_extraction(input_path, output_dir, save_as="py")

    # logger.info("Extracting documentation markdown...")
    # run_text_extraction(input_path, output_dir, save_as="md")
