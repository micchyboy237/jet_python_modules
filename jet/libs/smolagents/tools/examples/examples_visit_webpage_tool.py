import shutil
from pathlib import Path

from jet.libs.smolagents.tools.visit_webpage_tool import VisitWebpageTool
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def demo_smart_no_query(tool: VisitWebpageTool, console: Console) -> None:
    """Demo: Default smart excerpts – no explicit query"""
    title = "Smart excerpts (default) – no query"
    url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    full_raw = False
    query = None

    _run_example(tool, console, title, url, full_raw, query)


def demo_smart_focused_query(tool: VisitWebpageTool, console: Console) -> None:
    """Demo: Smart excerpts with a specific focus query"""
    title = "Smart excerpts – focused query"
    url = "https://fastapi.tiangolo.com/"
    full_raw = False
    query = "key features and advantages of FastAPI"

    _run_example(tool, console, title, url, full_raw, query)


def demo_full_raw(tool: VisitWebpageTool, console: Console) -> None:
    """Demo: Full raw content (truncated)"""
    title = "Full raw content (truncated)"
    url = "https://www.python.org/downloads/"
    full_raw = True
    query = None

    _run_example(tool, console, title, url, full_raw, query)


def _run_example(
    tool: VisitWebpageTool,
    console: Console,
    title: str,
    url: str,
    full_raw: bool,
    query: str | None,
) -> None:
    """Shared logic for printing parameters, running forward(), showing preview."""
    console.print(f"\n[bold underline cyan]{title}[/]", justify="left")
    console.print(f"URL:    {url}")
    console.print(f"Mode:   {'Full raw' if full_raw else 'Smart excerpts'}")
    if query:
        console.print(f"Query:  {query}")

    with console.status("[bold green]Visiting webpage...", spinner="dots"):
        result = tool.forward(
            url=url,
            full_raw=full_raw,
            query=query,
        )

    preview_len = 380
    preview = result[:preview_len] + ("…" if len(result) > preview_len else "")

    console.print(
        Panel(
            Text(preview, style="grey70"),
            title=f"Result preview ({len(result):,} chars total)",
            border_style="dim blue",
            expand=False,
            padding=(1, 2),
        )
    )


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    # from jet.libs.smolagents.tools.visit_webpage_tool import VisitWebpageTool

    console = Console()
    console.rule("VisitWebpageTool – Quick Usage Examples", style="bold cyan")

    tool = VisitWebpageTool(
        verbose=True,
        # logs_dir="visit_logs_example",   # uncomment to keep logs
    )

    # Choose which demos to run (comment out the ones you don't want)
    # demo_smart_no_query(tool, console)
    demo_smart_focused_query(tool, console)
    # demo_full_raw(tool, console)

    console.rule("End of examples", style="dim")
