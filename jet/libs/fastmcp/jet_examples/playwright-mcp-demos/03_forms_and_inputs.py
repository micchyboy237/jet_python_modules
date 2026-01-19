# File: demos/03_forms_and_inputs.py
"""
Form filling demo
Tools: browser_type, browser_click, browser_file_upload, browser_press_key
"""

import asyncio
from pathlib import Path
from rich.console import Console
from rich.panel import Panel


console = Console()


async def main():
    console.print(Panel.fit(
        "[bold cyan]03 - Forms & Inputs[/bold cyan]\n"
        "type • click • file_upload • press_key",
        border_style="bright_blue"
    ))

    START_URL = "https://demoqa.com/automation-practice-form"
    console.print(f"\n[bold]Practice form:[/] [link={START_URL}]{START_URL}[/link]\n")

    from utils.base import get_client
    client = get_client()

    async with client:
        await client.call_tool("browser_navigate", {"url": START_URL})

        # -------------------------------------------------------------------------
        # Basic text inputs using browser_type (more reliable than fill_form here)
        console.print("[yellow]Filling basic text fields...[/yellow]")

        browser_type_first_name = await client.call_tool("browser_type", {
            "element": "#firstName",
            "text": "Robert",
            "slowly": False
        })

        browser_type_last_name = await client.call_tool("browser_type", {
            "element": "Last Name",
            "text": "Smith",
            "slowly": False
        })

        await client.call_tool("browser_type", {
            "element": "Email input field",
            "ref": "",
            "text": "robert.smith@example.com",
            "slowly": False
        })

        await client.call_tool("browser_type", {
            "element": "Mobile number field",
            "ref": "",
            "text": "5551234567",
            "slowly": False
        })

        console.print("[green]✓ Basic text fields filled[/green]")

        # Gender selection (radio button)
        console.print("\n[yellow]Selecting Gender → Male...[/yellow]")
        await client.call_tool("browser_click", {
            "element": "Male radio button",
            "ref": ""
        })
        console.print("[green]✓ Gender selected[/green]")

        # State → NCR (type + Enter)
        console.print("\n[yellow]Selecting State → NCR...[/yellow]")
        await client.call_tool("browser_click", {
            "element": "State dropdown",
            "ref": ""
        })
        await client.call_tool("browser_type", {
            "element": "State search input",
            "ref": "",
            "text": "NCR"
        })
        await client.call_tool("browser_press_key", {"key": "Enter"})
        await asyncio.sleep(0.8)

        # City → Delhi (same pattern)
        console.print("[yellow]Selecting City → Delhi...[/yellow]")
        await client.call_tool("browser_click", {
            "element": "City dropdown",
            "ref": ""
        })
        await client.call_tool("browser_type", {
            "element": "City search input",
            "ref": "",
            "text": "Delhi"
        })
        await client.call_tool("browser_press_key", {"key": "Enter"})
        console.print("[green]✓ State & City selected[/green]")

        # File upload – first trigger the input
        picture_path = Path.home() / "Pictures" / "profile.jpg"
        if picture_path.exists():
            console.print(f"\n[yellow]Uploading picture: {picture_path.name}[/yellow]")
            # Click the label or the hidden input to trigger file chooser
            await client.call_tool("browser_click", {
                "element": "Picture upload button or label",
                "ref": ""
            })
            await asyncio.sleep(0.5)  # give time for file dialog
            await client.call_tool("browser_file_upload", {
                "paths": [str(picture_path)]
            })
            console.print("[green]✓ Picture uploaded[/green]")
        else:
            console.print("\n[orange1]Skipping file upload — no profile.jpg found[/orange1]")

        # Optional: fill address
        console.print("\n[yellow]Filling Current Address...[/yellow]")
        await client.call_tool("browser_type", {
            "element": "Current Address textarea",
            "ref": "",
            "text": "123 Demo Street, Makati City"
        })
        console.print("[green]✓ Address filled[/green]")

    console.print("\n[bold bright_green]Forms demo finished![/bold bright_green]\n")


if __name__ == "__main__":
    asyncio.run(main())