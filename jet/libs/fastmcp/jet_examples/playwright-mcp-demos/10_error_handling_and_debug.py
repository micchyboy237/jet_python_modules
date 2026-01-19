# File: demos/10_error_handling_and_debug.py
"""
Error handling & debugging utilities demo
Tools: browser_console_messages, browser_handle_dialog, browser_install (conceptual)
"""

import asyncio
from rich.console import Console
from rich.panel import Panel


console = Console()


async def main():
    console.print(Panel.fit(
        "[bold cyan]10 - Error Handling & Debugging Utilities[/bold cyan]",
        border_style="bright_blue"
    ))

    from utils.base import get_client
    client = get_client()

    async with client:
        # Console messages example
        console.print("[yellow]Loading page that might log to console...[/yellow]")
        await client.call_tool("browser_navigate", {
            "url": "https://the-internet.herokuapp.com/javascript_alerts"
        })

        await asyncio.sleep(1.0)

        console.print("\n[yellow]Console messages after page load:[/yellow]")
        messages = await client.call_tool("browser_console_messages")
        if messages and messages.get("messages"):
            for msg in messages["messages"][:5]:  # limit output
                console.print(f"[{msg.get('level','?')}] {msg.get('text','—')}")
        else:
            console.print("[dim]No console messages detected[/dim]")

        # Alert handling example
        console.print("\n[yellow]Triggering JS Alert...[/yellow]")
        await client.call_tool("browser_click", {
            "element": "Click for JS Alert button"
        })

        await asyncio.sleep(0.8)

        console.print("[yellow]Accepting alert...[/yellow]")
        await client.call_tool("browser_handle_dialog", {
            "accept": True
        })
        console.print("[green]✓ Alert accepted[/green]")

    console.print("\n[bold bright_green]Debug & error handling demo finished![/bold bright_green]\n")


if __name__ == "__main__":
    asyncio.run(main())