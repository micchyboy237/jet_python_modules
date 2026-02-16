import re
import shutil
from pathlib import Path

from jet.libs.smolagents.custom_models import OpenAIModel
from jet.libs.smolagents.tools.local_file_read_tool import LocalFileReadTool
from jet.libs.smolagents.tools.local_file_search_tool import LocalFileSearchTool
from jet.libs.smolagents.tools.local_file_write_tool import LocalFileWriteTool
from smolagents import LogLevel, ToolCallingAgent

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
WORK_DIR = OUTPUT_DIR / "code"
WORK_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Local LLM configuration
# ---------------------------------------------------------------------------

model = OpenAIModel(
    model_id="qwen3-instruct-2507:4b",
    temperature=0.2,
    max_tokens=12000,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

search_tool = LocalFileSearchTool()
read_tool = LocalFileReadTool()
write_tool = LocalFileWriteTool(work_dir=str(WORK_DIR))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

agent = ToolCallingAgent(
    tools=[search_tool, read_tool, write_tool],
    model=model,
    add_base_tools=False,
    verbosity_level=LogLevel.DEBUG,
)


# ---------------------------------------------------------------------------
# Markdown python extractor
# ---------------------------------------------------------------------------

PYTHON_BLOCK_RE = re.compile(
    r"```python\s*(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_python_blocks(markdown: str) -> list[str]:
    return [block.strip() for block in PYTHON_BLOCK_RE.findall(markdown)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Extract Python code blocks from Markdown files using a tool-driven agent."
    )
    parser.add_argument(
        "task",
        type=str,
        help="Instruction for the agent (quoted string).",
    )
    parser.add_argument(
        "base_dir", type=str, help="Base directory to search for markdown files."
    )
    parser.add_argument(
        "target",
        nargs="+",
        help="One or more target markdown files (filenames or globs, space separated).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    base_dir = args.base_dir
    target_files = args.target
    task = args.task

    # Adapt task so agent is told which files are being searched.
    task_prompt = (
        f"Search these file(s) under this directory:\n"
        f"Base directory: {base_dir}\n"
        f"File(s): {', '.join(target_files)}\n\n"
        f"{task}\n"
        "For all specified Markdown files, generate a single Python file per document containing full working reusable code with usage examples found in python code blocks."
    )

    search_result = agent.run(task_prompt)
    print(search_result)

    # Deterministic extraction + writing
    base_path = Path(base_dir).resolve()
    output_root = Path(write_tool.work_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Find matching .md files based on the provided targets
    md_files_to_extract = set()
    for t in target_files:
        # Support full paths (if provided), globs, and filenames
        local_matches = (
            list(base_path.rglob(t)) if not Path(t).is_absolute() else [Path(t)]
        )
        for match in local_matches:
            # Only include .md files (in case globs are too broad)
            if match.is_file() and match.suffix == ".md":
                md_files_to_extract.add(match.resolve())

    for md_file in md_files_to_extract:
        try:
            text = md_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        blocks = extract_python_blocks(text)
        if not blocks:
            continue

        stem = md_file.stem.replace("-", "_").replace(" ", "_")
        combined_code: list[str] = []

        for idx, code in enumerate(blocks, start=1):
            combined_code.append(
                f"# ------------------------------------------------------------------\n"
                f"# Source: {md_file.name} | Block {idx}\n"
                f"# ------------------------------------------------------------------\n"
                f"{code}\n"
            )

        # Step 1: Compute relative directory
        try:
            relative_dir = md_file.parent.relative_to(base_path)
        except ValueError:
            # md_file is not under base_path
            relative_dir = Path()

        # Step 2: Build output path using that directory
        relative_path = relative_dir / f"{stem}_example.py"

        # Step 3: Pass it to the write tool
        result = write_tool.forward(
            relative_path=str(relative_path),
            content="\n".join(combined_code),
        )

        print(result)

    print(f"\nPython files generated under: {output_root}")
