"""
TEMPLATE – copy this for new numbered demos
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from utils.starting_points import STARTING_POINTS, DEFAULT_START
from utils.demo_helpers import step, navigate_to
from utils.base import get_client

console = Console()

async def main() -> None:
    console.print(Panel.fit(
        "[bold cyan]DEMO TITLE[/bold cyan]\nWhich tools are shown here",
        border_style="bright_blue"
    ))

    start_cfg = STARTING_POINTS[DEFAULT_START]
    console.print(f"\n[bold]Demo site:[/] {start_cfg['description']}")
    console.print(f" URL: [link={start_cfg['url']}]{start_cfg['url']}[/link]\n")

    client = get_client()      # ← adjust if needed

    async with client:
        await navigate_to(client, start_cfg["url"])

        # ── Add your demo steps here ────────────────────────────────────────
        await step(client, "Example step", lambda: None, sleep_after=0.8)

        console.print("\n[bold green]✓ Demo finished[/bold green]")

if __name__ == "__main__":
    asyncio.run(main())
