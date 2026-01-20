from pathlib import Path
import shutil
from typing import List
import re

from smolagents import OpenAIModel, ToolCallingAgent, LogLevel

from jet.libs.smolagents.tools.local_file_search_tool import LocalFileSearchTool
from jet.libs.smolagents.tools.local_file_write_tool import LocalFileWriteTool

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
WORK_DIR = OUTPUT_DIR / "code"
WORK_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Local LLM configuration
# ---------------------------------------------------------------------------

model = OpenAIModel(
    model_id="local-model",
    api_base="http://shawn-pc.local:8080/v1",
    api_key="not-needed",
    temperature=0.2,
    max_tokens=2048,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

search_tool = LocalFileSearchTool()
write_tool = LocalFileWriteTool(work_dir=str(WORK_DIR))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

agent = ToolCallingAgent(
    tools=[search_tool, write_tool],
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


def extract_python_blocks(markdown: str) -> List[str]:
    return [block.strip() for block in PYTHON_BLOCK_RE.findall(markdown)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base_dir = (
        "/Users/jethroestrada/Desktop/External_Projects/AI/"
        "repo-libs/smolagents/docs/source/en"
    )

    # Ask agent to discover files (tool-driven)
    search_result = agent.run(
        f"""
        Find all Markdown files under this directory
        that contain Python code examples.

        Base directory: {base_dir}
        File pattern: **/*.md
        The files must contain ```python code blocks.
        Limit results to 50 files.
        """
    )

    print(search_result)

    # Deterministic extraction + writing
    base_path = Path(base_dir).resolve()
    output_root = Path(write_tool.work_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for md_file in base_path.rglob("*.md"):
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
            relative_dir = Path()  # write directly into the root

        # Step 2: Build output path using that directory
        relative_path = (relative_dir / f"{stem}_example.py")

        # Step 3: Pass it to the write tool
        result = write_tool.forward(
            relative_path=str(relative_path),
            content="\n".join(combined_code),
        )

        print(result)

    print(f"\nPython files generated under: {output_root}")
