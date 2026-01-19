# File: demos/05_content_extraction.py
"""
Content extraction & inspection demo
Tools: browser_snapshot, browser_evaluate, browser_console_messages
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.pretty import pprint


console = Console()


async def main():
    console.print(Panel.fit(
        "[bold cyan]05 - Content Extraction & Inspection[/bold cyan]",
        border_style="bright_blue"
    ))

    START_URL = "https://demoqa.com/text-box"
    console.print(f"\n[bold]Starting page:[/] [link={START_URL}]{START_URL}[/link]\n")

    from utils.base import get_client
    client = get_client()

    async with client:
        await client.call_tool("browser_navigate", {"url": START_URL})
        console.print("[green]✓ Text Box page loaded[/green]\n")

        # Get page title and current URL via evaluate
        console.print("[yellow]Extracting title and URL via evaluate...[/yellow]")
        title_result = await client.call_tool("browser_evaluate", {
            "function": "() => document.title"
        })
        url_result = await client.call_tool("browser_evaluate", {
            "function": "() => window.location.href"
        })

        console.print(f"[green]Title:[/] {title_result.get('result', '—')}")
        console.print(f"[green]Current URL:[/] {url_result.get('result', '—')}\n")

        # Get full name input value
        console.print("[yellow]Getting value of Full Name field...[/yellow]")
        value_result = await client.call_tool("browser_evaluate", {
            "element": "Full Name input field",
            "function": "(el) => el.value"
        })
        console.print(f"[green]Full Name field value:[/] {value_result.get('result', '—')}\n")

        # Take accessibility snapshot
        console.print("[yellow]Creating accessibility snapshot...[/yellow]")
        await client.call_tool("browser_snapshot", {
            "filename": "snapshot-textbox.md"  # optional - can be omitted
        })
        console.print("[green]✓ Accessibility snapshot saved[/green]")

        # Show some console messages (probably few on this page)
        console.print("\n[yellow]Reading console messages...[/yellow]")
        messages = await client.call_tool("browser_console_messages", {"level": "warning"})
        if messages and messages.get("messages"):
            console.print("[dim]Console messages (warnings+):[/dim]")
            pprint(messages["messages"], max_length=6)
        else:
            console.print("[dim]No console warnings/errors found[/dim]")

    console.print("\n[bold bright_green]Content extraction demo finished![/bold bright_green]\n")


if __name__ == "__main__":
    asyncio.run(main())