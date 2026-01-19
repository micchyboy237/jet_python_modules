# File: demos/06and_snapshots.py
"""
Element snapshots
Tools: browser_snapshot
"""

import asyncio
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from utils.base import get_client, get_output_dir


console = Console()


async def main():
    client = get_client()
    output_dir = get_output_dir()

    console.print(Panel.fit(
        "[bold cyan]06 - Element snapshots [/bold cyan]",
        border_style="bright_blue"
    ))

    START_URL = "https://demoqa.com/automation-practice-form"
    console.print(f"\n[bold]Target page:[/] [link={START_URL}]{START_URL}[/link]\n")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    async with client:
        await client.call_tool("browser_navigate", {"url": START_URL})

        # Accessibility snapshot
        console.print("\n[yellow]Creating accessibility snapshot...[/yellow]")
        await client.call_tool("browser_snapshot", {
            "filename": f"{output_dir}/snapshot-practice-form-{timestamp}.md"
        })
        console.print("[green]✓ Accessibility snapshot saved[/green]")

    console.print("\n[bold bright_green]Screenshots demo completed![/bold bright_green]\n")


if __name__ == "__main__":
    asyncio.run(main())