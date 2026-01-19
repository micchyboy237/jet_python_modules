# File: demos/06_screenshots_and_snapshots.py
"""
Screenshot & visual capture demo
Tools: browser_take_screenshot, browser_snapshot
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from utils.base import get_client, get_output_dir

console = Console()


async def main():
    client = get_client()
    output_dir = get_output_dir()

    console.print(Panel.fit(
        "[bold cyan]06 - Screenshots & Visual Captures[/bold cyan]",
        border_style="bright_blue"
    ))

    START_URL = "https://demoqa.com/automation-practice-form"
    console.print(f"\n[bold]Target page:[/] [link={START_URL}]{START_URL}[/link]\n")

    async with client:
        await client.call_tool("browser_navigate", {"url": START_URL})

        # Accessibility snapshot
        console.print("\n[yellow]Creating accessibility snapshot...[/yellow]")
        await client.call_tool("browser_snapshot", {
            "filename": f"{output_dir}/snapshot-practice-form.md"
        })
        console.print("[green]✓ Accessibility snapshot saved[/green]")

        # Full page screenshot
        console.print("[yellow]Taking full page screenshot...[/yellow]")
        await client.call_tool("browser_take_screenshot", {
            "type": "png",
            "filename": f"{output_dir}/screenshot-full.png",
            "fullPage": True
        })
        console.print("[green]✓ Full page screenshot saved[/green]")

        # Full page screenshot
        console.print("\n[yellow]Taking full page screenshot...[/yellow]")
        await client.call_tool("browser_take_screenshot", {
            "type": "png",
            "filename": f"{output_dir}/screenshot-full-page.png",
            "fullPage": True
        })
        console.print("[green]✓ Full page screenshot saved[/green]")

        # Viewport screenshot
        console.print("\n[yellow]Taking viewport screenshot...[/yellow]")
        await client.call_tool("browser_take_screenshot", {
            "type": "png",
            "filename": f"{output_dir}/screenshot-viewport.png",
            "fullPage": False
        })
        console.print("[green]✓ Viewport screenshot saved[/green]")

        # # Element screenshot - Submit button
        # console.print("\n[yellow]Taking screenshot of Submit button...[/yellow]")
        # await client.call_tool("browser_take_screenshot", {
        #     "element": "Submit button at the bottom of the form",
        #     "ref": "", # Use LLM browser_snapshot tool to get submit ref id
        #     "type": "png",
        #     "filename": f"{output_dir}/screenshot-submit-btn.png"
        # })
        # console.print("[green]✓ Element screenshot saved[/green]")

    console.print("\n[bold bright_green]Screenshots demo completed![/bold bright_green]\n")


if __name__ == "__main__":
    asyncio.run(main())