import asyncio
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from jet.agents.llama_cpp.rag_web_crawler_agent import run_webswarm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ─── Configuration ────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
QUERY = "What are the deployment options for LangGraph Platform?"
ROOT_URL = "https://langchain-ai.github.io/langgraph/"


# ─── Dual Output Writer ───────────────────────────────────────────────────────
class TeeWriter:
    """Writes to both terminal (via Rich) and a log file simultaneously."""

    def __init__(self, term_console: Console, log_file: TextIO):
        self._term = term_console
        self._log = log_file

    def write(self, text: str) -> int:
        # Write raw text to log file
        self._log.write(text)
        self._log.flush()
        # Write to terminal with Rich (handles ANSI/formatting)
        if text.strip():
            self._term.print(text, end="", highlight=False)
        elif text:
            # Preserve whitespace/newlines
            self._term.file.write(text)
            self._term.file.flush()
        return len(text)

    def flush(self):
        self._log.flush()
        self._term.file.flush()


# ─── Artifact Persistence ─────────────────────────────────────────────────────
def save_artifacts(result: dict, output_dir: Path) -> list[Path]:
    """Persist all run artifacts and return list of saved file paths."""
    saved_files: list[Path] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    # 1. Answer (Markdown)
    answer_path = output_dir / "answer.md"
    answer_content = (
        f"# RAG Web Crawler Result\n\n"
        f"**Query:** {QUERY}\n"
        f"**Root:** {ROOT_URL}\n"
        f"**Generated:** {timestamp}\n\n---\n\n"
        f"{result.get('answer', 'No answer generated.')}\n"
    )
    answer_path.write_text(answer_content, encoding="utf-8")
    saved_files.append(answer_path)

    # 2. Knowledge Base (Full JSON with content & scores)
    kb_path = output_dir / "knowledge_base.json"
    # result now includes 'knowledge_base' from enriched SwarmResult
    kb_entries = result.get("knowledge_base", [])
    kb_data = [
        {
            "url": entry.get("url", ""),
            "score": round(entry.get("score", 0.0), 4),
            "raw_score": round(entry.get("raw_score", 0.0), 4),
            "original_chars": entry.get("original_chars", 0),
            "truncated_chars": entry.get("truncated_chars", 0),
            "content_preview": entry.get("content", "")[:500],
        }
        for entry in kb_entries
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
        "kb_size": len(kb_entries),
        "timestamp": timestamp,
        "output_dir": str(output_dir.resolve()),
    }
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    saved_files.append(meta_path)

    # 4. Sources Summary
    sources_path = output_dir / "sources.txt"
    sources = result.get("sources", [])
    sources_lines = [f"[{i + 1}] {url}" for i, url in enumerate(sources)]
    sources_path.write_text(
        "\n".join(sources_lines) or "No sources retrieved.", encoding="utf-8"
    )
    saved_files.append(sources_path)

    # 5. Log file path (already written during execution)
    log_path = output_dir / "agent_trace.log"
    saved_files.append(log_path)

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
        abs_uri = f"file://{f.resolve()}"
        table.add_row(f"[link={abs_uri}]{f.name}[/link]", str(rel_path), size_str)

    Console().print()
    Console().print(table)
    Console().print()


# ─── Main Entry Point ─────────────────────────────────────────────────────────
async def main():
    # Prepare output directory
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_path = OUTPUT_DIR / "agent_trace.log"
    log_file = log_path.open("w", encoding="utf-8")
    term_console = Console(
        stderr=True
    )  # Use stderr for terminal to avoid capture issues

    tee = TeeWriter(term_console, log_file)
    original_stdout = sys.stdout

    try:
        # Redirect stdout so agent's print() calls go through TeeWriter
        sys.stdout = tee  # type: ignore[assignment]

        term_console.print(
            Panel(
                f"[bold]Query:[/bold] {QUERY}\n[bold]Root:[/bold] {ROOT_URL}",
                title="🕷️ RAG Web Crawler Agent",
                border_style="blue",
            )
        )

        result = await run_webswarm(query=QUERY, root_url=ROOT_URL)

    except Exception as e:
        term_console.print(f"\n[bold red]❌ Agent failed:[/bold red] {e}")
        raise
    finally:
        sys.stdout = original_stdout
        log_file.close()

    # Save artifacts
    saved_files = save_artifacts(result, OUTPUT_DIR)

    # Display completion summary
    term_console.print(
        Panel(
            f"[bold green]✓ Completed[/bold green] in {result.get('iterations', '?')} iterations | "
            f"{result.get('pages_visited', '?')} pages visited | "
            f"{len(result.get('sources', []))} sources",
            border_style="green",
        )
    )

    display_resource_links(saved_files, OUTPUT_DIR)


if __name__ == "__main__":
    asyncio.run(main())
