# File: demos/02_basic_navigation.py
"""
Basic browser control & navigation demo
Tools used: browser_navigate, browser_resize, browser_navigate_back, browser_close
"""

import asyncio
from rich.console import Console
from rich.panel import Panel


console = Console()


async def main():
    console.print(Panel.fit(
        "[bold cyan]02 - Basic Navigation & Browser Control[/bold cyan]",
        border_style="bright_blue", padding=(1, 2)
    ))

    START_URL = "https://demoqa.com/automation-practice-form"
    console.print(f"\n[bold]Target page:[/] [link={START_URL}]{START_URL}[/link]\n")

    from utils.base import get_client
    client = get_client()

    async with client:
        with console.status("[bold green]Navigating...[/bold green]", spinner="dots"):
            await client.call_tool("browser_navigate", {"url": START_URL})
            await asyncio.sleep(1.8)

        console.print("[green]✓ Page loaded[/green]")

        console.print("\n[yellow]→ Resizing browser window...[/yellow]")
        await client.call_tool("browser_resize", {"width": 1280, "height": 960})
        await asyncio.sleep(1.2)
        console.print("[green]✓ Resized to 1280×960[/green]")

        console.print("\n[yellow]→ Going to another page...[/yellow]")
        await client.call_tool("browser_navigate", {"url": "https://demoqa.com/text-box"})
        await asyncio.sleep(1.5)
        console.print("[green]✓ Navigated to Text Box page[/green]")

        console.print("\n[yellow]↩ Going back...[/yellow]")
        await client.call_tool("browser_navigate_back")
        await asyncio.sleep(1.3)
        console.print("[green]✓ Returned to practice form[/green]")

    console.print("\n[bold bright_green]Basic navigation demo finished![/bold bright_green]\n")


if __name__ == "__main__":
    asyncio.run(main())