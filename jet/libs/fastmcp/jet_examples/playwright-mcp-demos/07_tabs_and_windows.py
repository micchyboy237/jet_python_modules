# File: demos/07_tabs_and_windows.py
"""
Tab & window management demo
Tool: browser_tabs
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


console = Console()


async def main():
    console.print(Panel.fit(
        "[bold cyan]07 - Tabs & Window Management[/bold cyan]",
        border_style="bright_blue"
    ))

    START_URL = "https://the-internet.herokuapp.com/windows"
    console.print(f"\n[bold]Starting page:[/] [link={START_URL}]{START_URL}[/link]\n")

    from utils.base import get_client
    client = get_client()

    async with client:
        await client.call_tool("browser_navigate", {"url": START_URL})
        console.print("[green]✓ Windows demo page loaded[/green]\n")

        # List current tabs
        console.print("[yellow]Current tabs:[/yellow]")
        tabs = await client.call_tool("browser_tabs", {"action": "list"})
        table = Table("Index", "URL", "Active")
        for i, tab in enumerate(tabs.get("tabs", []), 1):
            table.add_row(str(i), tab.get("url", "—"), "Yes" if tab.get("active") else "—")
        console.print(table)
        console.print("")

        # Open new tab (click link that opens in new window)
        console.print("[yellow]Opening new tab/window via click...[/yellow]")
        await client.call_tool("browser_click", {
            "element": "Click Here link that opens new window",
            "doubleClick": False
        })
        await asyncio.sleep(2.0)

        # List tabs again
        console.print("[yellow]Tabs after opening new window:[/yellow]")
        tabs_after = await client.call_tool("browser_tabs", {"action": "list"})
        table = Table("Index", "URL", "Active")
        for i, tab in enumerate(tabs_after.get("tabs", []), 1):
            table.add_row(str(i), tab.get("url", "—"), "Yes" if tab.get("active") else "—")
        console.print(table)

        # Switch to first tab (usually original)
        console.print("\n[yellow]Switching back to first tab...[/yellow]")
        await client.call_tool("browser_tabs", {
            "action": "select",
            "index": 1
        })
        console.print("[green]✓ Switched to tab 1[/green]")

    console.print("\n[bold bright_green]Tabs management demo finished![/bold bright_green]\n")


if __name__ == "__main__":
    asyncio.run(main())