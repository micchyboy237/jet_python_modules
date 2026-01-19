# File: demos/04_waiting_and_content.py
"""
Waiting & content extraction demo
Tools: browser_wait_for, browser_snapshot, browser_evaluate
"""

import asyncio
from rich.console import Console
from rich.panel import Panel


console = Console()


async def main():
    console.print(Panel.fit(
        "[bold cyan]04 - Waiting & Content Extraction[/bold cyan]",
        border_style="bright_blue"
    ))

    from utils.base import get_client
    client = get_client()

    async with client:
        await client.call_tool("browser_navigate", {
            "url": "https://demoqa.com/dynamic-properties"
        })
        console.print("[green]✓ Dynamic Properties page loaded[/green]\n")

        console.print("[yellow]Waiting for visible text button to appear (max 10s)...[/yellow]")
        try:
            await client.call_tool("browser_wait_for", {
                "text": "Visible After 5 seconds",
                "time": 12  # safety margin
            })
            console.print("[green]✓ Text appeared![/green]")
        except Exception as e:
            console.print(f"[red]Timeout or error: {e}[/red]")

        # Get page title via evaluate
        console.print("\n[yellow]Getting page title via evaluate...[/yellow]")
        result = await client.call_tool("browser_evaluate", {
            "function": "() => document.title"
        })
        console.print(f"[green]Page title:[/] {result.get('result', '—')}")

        # Take accessibility snapshot
        console.print("\n[yellow]Taking accessibility snapshot...[/yellow]")
        await client.call_tool("browser_snapshot", {
            "filename": "snapshot-dynamic-props.md"  # optional
        })
        console.print("[green]✓ Snapshot captured[/green]")

    console.print("\n[bold bright_green]Waiting & content demo completed![/bold bright_green]\n")


if __name__ == "__main__":
    asyncio.run(main())