# examples_visit_webpage_tool.py
import shutil
from pathlib import Path

from jet.libs.smolagents.tools.visit_webpage_tool import VisitWebpageTool
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Prepare output directory
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

console = Console()


def save_result(name: str, content: str):
    """Helper to save demo output to file"""
    path = OUTPUT_DIR / f"{name}.md"
    path.write_text(content, encoding="utf-8")
    console.print(f"[green]Saved:[/green] {path}")


# ────────────────────────────────────────────────
# Demo functions
# ────────────────────────────────────────────────


def demo_basic_usage():
    """Demo 1: Basic usage – default smart excerpts"""
    tool = VisitWebpageTool(verbose=True)

    result = tool.forward(
        url="https://example.com",
        full_raw=False,
        query=None,  # auto-inferred
    )

    console.print(
        Panel.fit(
            Text(result, style="white"),
            title="Demo 1: Basic usage (smart excerpts)",
            border_style="bright_blue",
        )
    )

    save_result("demo_01_basic", result)


def demo_with_specific_query():
    """Demo 2: With a specific focus query"""
    tool = VisitWebpageTool(verbose=True)

    result = tool.forward(
        url="https://en.wikipedia.org/wiki/Python_(programming_language)",
        query="What are the main advantages of Python in 2025?",
        full_raw=False,
    )

    console.print(
        Panel.fit(
            Text(result, style="white"),
            title="Demo 2: Focused query",
            border_style="green",
        )
    )

    save_result("demo_02_focused_query", result)


def demo_full_raw_content():
    """Demo 3: Full raw markdown content (truncated)"""
    tool = VisitWebpageTool(verbose=True, max_output_length=4000)

    result = tool.forward(url="https://news.ycombinator.com", full_raw=True)

    console.print(
        Panel.fit(
            Text(result[:600] + "\n...\n(truncated)", style="grey"),
            title="Demo 3: Full raw content (truncated)",
            border_style="yellow",
        )
    )

    save_result("demo_03_full_raw", result)


def demo_custom_chunk_size_and_overlap():
    """Demo 4: Custom chunk size and overlap"""
    tool = VisitWebpageTool(
        verbose=True, chunk_target_tokens=300, chunk_overlap_tokens=80, top_k=6
    )

    result = tool.forward(
        url="https://www.python.org/about/gettingstarted/",
        query="How to get started with Python in 2025",
        full_raw=False,
    )

    console.print(
        Panel.fit(
            Text(result, style="white"),
            title="Demo 4: Custom chunk size & overlap",
            border_style="magenta",
        )
    )

    save_result("demo_04_custom_chunk", result)


def demo_relative_categories():
    """Demo 5: Using relative category mode (more robust across page sizes)"""
    tool = VisitWebpageTool(verbose=True)

    # Override with relative category config
    result = tool.forward(
        url="https://fastapi.tiangolo.com/",
        query="What are the main features of FastAPI?",
        full_raw=False,
    )

    console.print(
        Panel.fit(
            Text(result, style="white"),
            title="Demo 5: Relative category mode",
            border_style="cyan",
        )
    )

    save_result("demo_05_relative_categories", result)


def demo_error_handling():
    """Demo 6: Error handling – invalid URL"""
    tool = VisitWebpageTool(verbose=True)

    result = tool.forward(
        url="https://this-is-not-a-real-website-1234567890.com", full_raw=False
    )

    console.print(
        Panel.fit(
            Text(result, style="red"),
            title="Demo 6: Error handling (invalid URL)",
            border_style="red",
        )
    )

    save_result("demo_06_error_handling", result)


def run_all_demos():
    """Run all demo functions one after another"""
    console.print("\n[bold green]Running all VisitWebpageTool demos...[/bold green]\n")

    demos = [
        demo_basic_usage,
        demo_with_specific_query,
        demo_full_raw_content,
        demo_custom_chunk_size_and_overlap,
        demo_relative_categories,
        demo_error_handling,
    ]

    for demo in demos:
        console.rule(f"[bold]{demo.__doc__ or demo.__name__}[/bold]")
        try:
            demo()
        except Exception as e:
            console.print(f"[red]Error in {demo.__name__}: {e}[/red]")
        console.print("\n")


if __name__ == "__main__":
    run_all_demos()
    console.print("\n[bold green]All demos completed.[/bold green] Results saved in:")
    console.print(f"  → {OUTPUT_DIR}")
