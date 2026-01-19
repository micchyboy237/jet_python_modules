# File: demos/08_advanced_interaction.py
"""
Advanced interaction demo
Tools: browser_drag, browser_click (with modifiers), browser_select_option
"""

import asyncio
from rich.console import Console
from rich.panel import Panel


console = Console()


async def main():
    console.print(Panel.fit(
        "[bold cyan]08 - Advanced Interactions[/bold cyan]\n"
        "drag • click with modifiers • complex selections",
        border_style="bright_blue"
    ))

    START_URL = "https://the-internet.herokuapp.com/drag_and_drop"
    console.print(f"\n[bold]Drag & Drop demo page:[/] [link={START_URL}]{START_URL}[/link]\n")

    from utils.base import get_client
    client = get_client()

    async with client:
        await client.call_tool("browser_navigate", {"url": START_URL})
        console.print("[green]✓ Drag & Drop page loaded[/green]\n")

        console.print("[yellow]Performing drag & drop A → B...[/yellow]")
        await client.call_tool("browser_drag", {
            "startElement": "Column A",
            "endElement": "Column B",
        })
        console.print("[green]✓ Drag & drop completed[/green]")

        # Go to another page good for modifier clicks
        await client.call_tool("browser_navigate", {
            "url": "https://demoqa.com/buttons"
        })
        await asyncio.sleep(1.2)

        console.print("\n[yellow]Double clicking Double Click Me button...[/yellow]")
        await client.call_tool("browser_click", {
            "element": "Double Click Me button",
            "doubleClick": True
        })
        console.print("[green]✓ Double click performed[/green]")

    console.print("\n[bold bright_green]Advanced interactions demo finished![/bold bright_green]\n")


if __name__ == "__main__":
    asyncio.run(main())