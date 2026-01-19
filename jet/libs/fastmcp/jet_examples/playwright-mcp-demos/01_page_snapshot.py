# File: demos/01_page_snapshot.py
"""
Element snapshots
Tools: browser_snapshot
"""

import asyncio
from urllib.parse import urlencode
from rich.console import Console
from rich.panel import Panel
from utils.base import get_client, get_output_dir, BASE_OUTPUT_DIR
from utils.config_utils import extract_code_block_content, yaml_to_dict
from utils.page_utils import extract_all_references_ordered
from jet.file.utils import save_file


console = Console()


async def main():
    client = get_client()
    output_dir = get_output_dir()

    console.print(Panel.fit(
        "[bold cyan]01 - Element snapshots [/bold cyan]",
        border_style="bright_blue"
    ))

    # START_URL = "https://demoqa.com/automation-practice-form"

    query = "Top isekai anime today"
    search_params = {
        "q": query.strip(),
        # "format": "html",          # optional
        # "categories": "general",
    }
    query_string = urlencode(search_params)
    START_URL = f"http://jethros-macbook-air.local:8888?{query_string}"
    console.print(f"\n[bold]Target page:[/] [link={START_URL}]{START_URL}[/link]\n")

    async with client:
        browser_navigate_result = await client.call_tool("browser_navigate", {"url": START_URL})
        save_file(browser_navigate_result, f"{BASE_OUTPUT_DIR}/browser_navigate.json")

        browser_navigate_result_text = "\n\n\n".join(c.text for c in browser_navigate_result.content)
        yaml_config = extract_code_block_content(browser_navigate_result_text)
        config_dict = yaml_to_dict(yaml_config)
        page_info = extract_all_references_ordered(config_dict)
        save_file(page_info, f"{BASE_OUTPUT_DIR}/page_info.json")

        # Accessibility snapshot
        console.print("\n[yellow]Creating accessibility snapshot...[/yellow]")
        browser_snapshot_result = await client.call_tool("browser_snapshot", {
            "filename": f"{output_dir}/snapshot-practice-form.md"
        })
        save_file(browser_snapshot_result, f"{BASE_OUTPUT_DIR}/browser_snapshot.json")
        console.print("[green]✓ Accessibility snapshot saved[/green]")

        browser_snapshot_result_text = "\n\n\n".join(c.text for c in browser_snapshot_result.content)
        save_file(browser_snapshot_result_text, f"{BASE_OUTPUT_DIR}/browser_snapshot.md")

        console.print("[yellow]Taking full page screenshot...[/yellow]")
        await client.call_tool("browser_take_screenshot", {
            "type": "png",
            "filename": f"{output_dir}/screenshot-full.png",
            "fullPage": True
        })
        console.print("[green]✓ Full page screenshot saved[/green]")

    console.print("\n[bold bright_green]Screenshots demo completed![/bold bright_green]\n")


if __name__ == "__main__":
    asyncio.run(main())