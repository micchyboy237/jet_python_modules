# File: demos/02_interaction_basics.py
"""
Basic interaction demo
Tools: browser_click, browser_type, browser_hover, browser_press_key
"""

import asyncio
from rich.console import Console
from rich.panel import Panel


console = Console()


async def main():
    console.print(Panel.fit(
        "[bold cyan]02 - Basic Interactions[/bold cyan]\n"
        "click • type • hover • press_key",
        border_style="bright_blue"
    ))

    START_URL = "https://demoqa.com/automation-practice-form"
    console.print(f"\n[bold]Starting page:[/] [link={START_URL}]{START_URL}[/link]\n")

    from utils.base import get_client
    client = get_client()

    async with client:
        await client.call_tool("browser_navigate", {"url": START_URL})
        console.print("[green]✓ Practice form loaded[/green]\n")

        # Fill first name
        console.print("[yellow]Typing first name...[/yellow]")
        await client.call_tool("browser_type", {
            "element": "First Name input field",
            "ref": "",  # usually empty when using description
            "text": "John",
            "slowly": True
        })
        console.print("[green]✓ First name filled[/green]")

        # Hover over Submit button to see hover effect
        console.print("\n[yellow]Hovering over Submit button...[/yellow]")
        await client.call_tool("browser_hover", {
            "element": "Submit button at the bottom",
            "ref": ""
        })
        await asyncio.sleep(1.8)
        console.print("[green]✓ Hover performed[/green]")

        # Click radio button
        console.print("\n[yellow]Clicking Gender: Male radio...[/yellow]")
        await client.call_tool("browser_click", {
            "element": "Male radio button",
            "ref": "",
            "doubleClick": False
        })
        console.print("[green]✓ Male selected[/green]")

        # Press TAB key few times
        console.print("\n[yellow]Pressing TAB key 3 times...[/yellow]")
        for _ in range(3):
            await client.call_tool("browser_press_key", {"key": "Tab"})
            await asyncio.sleep(0.4)

        console.print("[green]✓ TAB keys pressed[/green]")

    console.print("\n[bold bright_green]Interaction basics demo completed![/bold bright_green]\n")


if __name__ == "__main__":
    asyncio.run(main())