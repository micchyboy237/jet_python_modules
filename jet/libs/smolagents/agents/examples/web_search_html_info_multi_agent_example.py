# jet_python_modules/jet/libs/smolagents/agents/examples/web_search_html_info_multi_agent_example.py
"""
Usage example / smoke test for WebSearchHTMLInfoMultiAgent
"""

import logging

# ─── Import the agent we just created ────────────────────────────────────────
from jet.libs.smolagents.agents.web_search_html_info_multi_agent import (
    create_web_html_info_agent,
)
from jet.libs.smolagents.tools.searxng_search_tool import SearXNGSearchTool
from rich.console import Console

console = Console()

# Adjust logging level if you want more/less verbosity
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web-html-example")


def main():
    # 1. Optional: customize the search tool if needed
    search_tool = SearXNGSearchTool(
        instance_url="http://searxng.local:8888",  # ← use your instance
        max_results=12,
        rate_limit=1.8,  # gentle on public instances
        timeout=12,
        verbose=True,
    )

    # 2. Create the full multi-agent pipeline
    agent = create_web_html_info_agent(
        search_tool=search_tool,
        # model_id="qwen3-instruct-2507:14b",   # ← uncomment to override default 4b
        max_pages=4,  # how many pages to deeply process
    )

    # 3. Example queries to try
    queries = [
        "latest developments in small language models 2026",
        "What is the current status of Grok 4 release?",
        "best open-source vector databases comparison 2026",
        "Python 3.13 new features performance impact",
    ]

    for i, query in enumerate(queries, 1):
        console.rule(f" Example {i} / {len(queries)} ", style="cyan")
        console.print(f"\n[bold]Query:[/] {query}\n", style="bold blue")

        try:
            answer = agent.run(query)

            # ─── Show result nicely ───────────────────────────────────────
            console.print("\n[green]✓ Pipeline completed[/green]\n")

            console.rule("Final Synthesized Answer", style="green")
            console.print(answer.strip(), markup=False)
            console.rule(style="green")

            # Optional: also log to file for later inspection
            from pathlib import Path

            out_path = Path("results") / f"result_{i}_{query[:40].replace(' ', '_')}.md"
            out_path.parent.mkdir(exist_ok=True)
            out_path.write_text(answer, encoding="utf-8")
            console.print(f"[dim]Saved → {out_path}[/dim]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user.[/yellow]")
            break
        except Exception as exc:
            console.print(f"\n[red]Error:[/] {exc.__class__.__name__}", style="red")
            logger.exception("Pipeline failed")
            console.print()


if __name__ == "__main__":
    console.print(
        "[bold magenta]Web → HTML → Structured Summary Multi-Agent Demo[/]\n",
        justify="center",
    )
    main()
    console.print("\n[dim]Finished.[/dim]\n")
