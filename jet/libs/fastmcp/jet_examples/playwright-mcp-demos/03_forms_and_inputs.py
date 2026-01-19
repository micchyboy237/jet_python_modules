# File: demos/03_forms_and_inputs.py
"""
Form filling demo
Tools: browser_fill_form, browser_file_upload, browser_select_option
"""

import asyncio
from pathlib import Path
from rich.console import Console
from rich.panel import Panel


console = Console()


async def main():
    console.print(Panel.fit(
        "[bold cyan]03 - Forms & Inputs[/bold cyan]\n"
        "fill_form • file_upload • select_option",
        border_style="bright_blue"
    ))

    START_URL = "https://demoqa.com/automation-practice-form"
    console.print(f"\n[bold]Practice form:[/] [link={START_URL}]{START_URL}[/link]\n")

    from utils.base import get_client
    client = get_client()

    async with client:
        await client.call_tool("browser_navigate", {"url": START_URL})

        # Multiple fields at once
        console.print("[yellow]Filling multiple fields with fill_form...[/yellow]")
        await client.call_tool("browser_fill_form", {
            "fields": [
                {"element": "First Name input", "value": "Robert"},
                {"element": "Last Name input", "value": "Smith"},
                {"element": "Email input field", "value": "robert.smith@example.com"},
                {"element": "Mobile number field", "value": "5551234567"},
            ]
        })
        console.print("[green]✓ Basic fields filled[/green]")

        # Select state and city
        console.print("\n[yellow]Selecting State → NCR...[/yellow]")
        await client.call_tool("browser_select_option", {
            "element": "State dropdown",
            "ref": "",
            "values": ["NCR"]
        })

        await asyncio.sleep(0.8)

        console.print("[yellow]Selecting City → Delhi...[/yellow]")
        await client.call_tool("browser_select_option", {
            "element": "City dropdown",
            "ref": "",
            "values": ["Delhi"]
        })
        console.print("[green]✓ State & City selected[/green]")

        # File upload example (you need to have this file)
        picture_path = Path.home() / "Pictures" / "profile.jpg"
        if picture_path.exists():
            console.print(f"\n[yellow]Uploading picture: {picture_path.name}[/yellow]")
            await client.call_tool("browser_file_upload", {
                "paths": [str(picture_path)]
            })
            console.print("[green]✓ Picture uploaded[/green]")
        else:
            console.print("\n[orange1]Skipping file upload — no profile.jpg found[/orange1]")

    console.print("\n[bold bright_green]Forms demo finished![/bold bright_green]\n")


if __name__ == "__main__":
    asyncio.run(main())