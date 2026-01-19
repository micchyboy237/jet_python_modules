"""
Basic Navigation & Browser Control Demo
Shows: navigate, back, resize, close
"""

import asyncio
from rich.console import Console
from rich.panel import Panel

from fastmcp import Client
from utils.starting_points import STARTING_POINTS, DEFAULT_START

console = Console()


async def main():
    console.print(Panel.fit(
        "[bold cyan]Basic Navigation Demo[/bold cyan]\n"
        "browser_navigate • browser_navigate_back • browser_resize • browser_close",
        border_style="bright_blue"
    ))

    # You can change starting point here
    start_config = STARTING_POINTS[DEFAULT_START]
    console.print(f"\n[bold]Starting point:[/] {start_config['description']}")
    console.print(f"         URL: [link={start_config['url']}]{start_config['url']}[/link]\n")

    client = Client.from_config_file()  # or your preferred way

    async with client:
        # 1. Navigate to starting page
        await client.call_tool("browser_navigate", {"url": start_config["url"]})
        console.print("[green]✓ Navigated to starting page[/green]")

        # 2. Resize window (nice for screenshots later)
        await client.call_tool("browser_resize", {"width": 1280, "height": 900})
        console.print("[green]✓ Browser resized to 1280×900[/green]")

        # 3. Let's navigate deeper (example links that usually exist)
        deeper_urls = [
            "https://www.saucedemo.com/inventory.html",  # after login
            "https://www.saucedemo.com/cart.html",
        ]

        for url in deeper_urls[:1]:  # just one for demo clarity
            console.print(f"→ Navigating to: [cyan]{url}[/cyan]")
            await client.call_tool("browser_navigate", {"url": url})
            await asyncio.sleep(1.2)  # let human see change

        # 4. Go back
        console.print("[yellow]↩ Going back...[/yellow]")
        await client.call_tool("browser_navigate_back")
        await asyncio.sleep(1.2)

        console.print("\n[bold green]Basic navigation demo finished![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())