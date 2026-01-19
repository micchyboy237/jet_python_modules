# File: demos/09_javascript_and_evaluation.py
"""
JavaScript execution & advanced evaluation demo
Tools: browser_run_code, browser_evaluate (complex)
"""

import asyncio
import os
from rich.console import Console
from rich.panel import Panel
from jet.utils.inspect_utils import get_entry_file_name
from utils.base import get_client, get_output_dir, BASE_OUTPUT_DIR
from jet.file.utils import save_file


console = Console()


async def fill_textarea(client, selector: str, value: str):
    # 2025-pattern, robust JS fill function
    escaped = value.replace("'", "\\'").replace("\n", "\\n")

    js = f"""
    (el) => {{
        if (!el) throw new Error('Element not found: ' + '{selector}');
        el.value = '{escaped}';
        el.dispatchEvent(new Event('input', {{bubbles: true}}));
        el.dispatchEvent(new Event('change', {{bubbles: true}}));
        el.dispatchEvent(new Event('blur', {{bubbles: true}}));
        return el.value;
    }}
    """

    return await client.call_tool("browser_evaluate", {
        "selector": selector,
        "function": js
    })


async def main():
    console.print(Panel.fit(
        f"[bold cyan]Start - {os.path.splitext(get_entry_file_name())[0]}[/bold cyan]",
        border_style="bright_blue"
    ))
    query = "Top isekai anime today"
    search_params = {
        "q": query.strip(),
        # "format": "html",          # optional
        # "categories": "general",
    }
    query_string = urlencode(search_params)
    START_URL = f"http://jethros-macbook-air.local:8888?{query_string}"
    console.print(f"\n[bold]Target page:[/] [link={START_URL}]{START_URL}[/link]\n")

    client = get_client()
    output_dir = get_output_dir()

    async with client:
        browser_navigate_result = await client.call_tool("browser_navigate", {"url": START_URL})
        save_file(browser_navigate_result, f"{BASE_OUTPUT_DIR}/browser_navigate.json")

        browser_navigate_result_text = "\n\n\n".join(c.text for c in browser_navigate_result.content)
        yaml_config = extract_code_block_content(browser_navigate_result_text)
        config_dict = yaml_to_dict(yaml_config)
        page_info = extract_all_references_ordered(config_dict)
        save_file(page_info, f"{BASE_OUTPUT_DIR}/page_info.json")

        # Simple evaluation
        count = await client.call_tool("browser_evaluate", {
            "function": "() => document.querySelectorAll('input').length"
        })
        count_text = str(count.content[0]) if count.content else "—"
        console.print(f"[green]Number of input fields: {count_text}[/green]\n")
        save_file(count_text, f"{BASE_OUTPUT_DIR}/count_text.md")

    console.print("\n[bold bright_green]JavaScript evaluation demo completed![/bold bright_green]\n")


if __name__ == "__main__":
    asyncio.run(main())