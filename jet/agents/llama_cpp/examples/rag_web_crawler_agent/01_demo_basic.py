import asyncio
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from jet.agents.llama_cpp.rag_web_crawler_agent import run_webswarm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ─── Configuration ────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
QUERY = "What are the deployment options for LangGraph Platform?"
ROOT_URL = "https://langchain-ai.github.io/langgraph/"

# ─── Rich Console Setup ───────────────────────────────────────────────────────
# Dual output: terminal + log file
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "agent_trace.log"
console = Console(file=LOG_FILE.open("w", encoding="utf-8"), highlight=True)
term_console = Console()  # For final resource links only


def save_artifacts(result: dict, output_dir: Path) -> list[Path]:
    """Persist all run artifacts and return list of saved file paths."""
    saved_files: list[Path] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    # 1. Answer (Markdown)
    answer_path = output_dir / "answer.md"
    answer_content = f"# RAG Web Crawler Result\n\n**Query:** {QUERY}\n**Root:** {ROOT_URL}\n**Generated:** {timestamp}\n\n---\n\n{result.get('answer', 'No answer generated.')}\n"
    answer_path.write_text(answer_content, encoding="utf-8")
    saved_files.append(answer_path)

    # 2. Knowledge Base (JSON) - Extract from result if available
    # Note: SwarmResult doesn't expose full KB, so we save what we have
    kb_path = output_dir / "knowledge_base.json"
    kb_data = [
        {"url": url, "relevance": "see sources.txt"}
        for url in result.get("sources", [])
    ]
    kb_path.write_text(
        json.dumps(kb_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    saved_files.append(kb_path)

    # 3. Run Metadata
    meta_path = output_dir / "run_metadata.json"
    metadata = {
        "query": QUERY,
        "root_url": ROOT_URL,
        "iterations": result.get("iterations", 0),
        "pages_visited": result.get("pages_visited", 0),
        "timestamp": timestamp,
        "output_dir": str(output_dir.resolve()),
    }
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    saved_files.append(meta_path)

    # 4. Sources Summary
    sources_path = output_dir / "sources.txt"
    sources_lines = [
        f"[{i + 1}] {url}" for i, url in enumerate(result.get("sources", []))
    ]
    sources_path.write_text(
        "\n".join(sources_lines) or "No sources retrieved.", encoding="utf-8"
    )
    saved_files.append(sources_path)

    # 5. Log file is already open/written via console
    saved_files.append(LOG_FILE)

    return saved_files


def display_resource_links(files: list[Path], base_dir: Path):
    """Display saved files as clickable resource links with base names."""
    table = Table(
        title="📁 Saved Artifacts", show_header=True, header_style="bold cyan"
    )
    table.add_column("File", style="green", no_wrap=True)
    table.add_column("Path", style="dim")
    table.add_column("Size", justify="right", style="yellow")

    for f in sorted(files):
        size = f.stat().st_size
        size_str = f"{size:,} B" if size < 1024 else f"{size / 1024:.1f} KB"
        rel_path = f.relative_to(base_dir)
        # Rich supports file:// links in compatible terminals
        link = f"file://{f.resolve()}"
        table.add_row(f"[link={link}]{f.name}[/link]", str(rel_path), size_str)

    term_console.print()
    term_console.print(table)
    term_console.print()


async def main():
    # Redirect stdout through Rich console during agent execution
    # This captures internal print() calls from rag_web_crawler_agent.py
    original_stdout = sys.stdout
    sys.stdout = console.file  # type: ignore[assignment]

    try:
        term_console.print(
            Panel(
                f"[bold]Query:[/bold] {QUERY}\n[bold]Root:[/bold] {ROOT_URL}",
                title="🕷️ RAG Web Crawler Agent",
                border_style="blue",
            )
        )

        result = await run_webswarm(query=QUERY, root_url=ROOT_URL)

    finally:
        sys.stdout = original_stdout
        console.file.close()  # Flush and close log file

    # Save artifacts
    saved_files = save_artifacts(result, OUTPUT_DIR)

    # Display summary
    term_console.print(
        Panel(
            f"[bold green]✓ Completed[/bold green] in {result.get('iterations', '?')} iterations | "
            f"{result.get('pages_visited', '?')} pages visited | "
            f"{len(result.get('sources', []))} sources",
            border_style="green",
        )
    )

    # Show resource links
    display_resource_links(saved_files, OUTPUT_DIR)


if __name__ == "__main__":
    asyncio.run(main())
