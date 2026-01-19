# File: demos/01_page_snapshot.py
"""
Element snapshots
Tools: browser_snapshot
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from utils.base import get_client, get_output_dir, BASE_OUTPUT_DIR
from jet.file.utils import save_file


console = Console()


async def main():
    client = get_client()
    output_dir = get_output_dir()

    console.print(Panel.fit(
        "[bold cyan]01 - Element snapshots [/bold cyan]",
        border_style="bright_blue"
    ))

    START_URL = "https://demoqa.com/automation-practice-form"
    console.print(f"\n[bold]Target page:[/] [link={START_URL}]{START_URL}[/link]\n")

    async with client:
        browser_navigate_result = await client.call_tool("browser_navigate", {"url": START_URL})
        save_file(browser_navigate_result, f"{BASE_OUTPUT_DIR}/browser_navigate.json")

        # Accessibility snapshot
        console.print("\n[yellow]Creating accessibility snapshot...[/yellow]")
        browser_snapshot_result = await client.call_tool("browser_snapshot", {
            "filename": f"{output_dir}/snapshot-practice-form.md"
        })
        save_file(browser_snapshot_result, f"{BASE_OUTPUT_DIR}/browser_snapshot.json")
        console.print("[green]✓ Accessibility snapshot saved[/green]")

        browser_snapshot_result_text = "\n\n\n".join(c.text for c in browser_snapshot_result.content)
        save_file(browser_snapshot_result_text, f"{BASE_OUTPUT_DIR}/browser_snapshot.md")

    console.print("\n[bold bright_green]Screenshots demo completed![/bold bright_green]\n")


if __name__ == "__main__":
    asyncio.run(main())