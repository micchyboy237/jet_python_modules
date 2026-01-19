# File: template/base.py
import asyncio
import os
from typing import Optional
from urllib.parse import urlencode
from rich.console import Console
from rich.panel import Panel
from utils.base import BASE_OUTPUT_DIR, get_client, get_tools
from utils.config_utils import extract_code_block_content, yaml_to_dict
from utils.page_utils import extract_all_references_ordered
from jet.utils.inspect_utils import get_entry_file_name
from jet.file.utils import save_file

console = Console()

async def search_query(query: str, url: Optional[str] = None) -> dict:
    console.print(Panel.fit(
        f"[bold cyan]Start - {os.path.splitext(get_entry_file_name())[0]}[/bold cyan]",
        border_style="bright_blue"
    ))
    
    search_params = {
        "q": query.strip(),
        # "format": "html",          # optional
        # "categories": "general",
    }
    query_string = urlencode(search_params)
    start_url = url or f"http://jethros-macbook-air.local:8888?{query_string}"
    console.print(f"\n[bold]Target page:[/] [link={start_url}]{start_url}[/link]\n")

    client = get_client()

    async with client:
        tools = await get_tools(client)
        save_file(tools, f"{BASE_OUTPUT_DIR}/tools.json")

        browser_navigate_result = await client.call_tool("browser_navigate", {"url": start_url})
        save_file(browser_navigate_result, f"{BASE_OUTPUT_DIR}/browser_navigate.json")

        browser_navigate_result_text = "\n\n\n".join(c.text for c in browser_navigate_result.content)
        yaml_config = extract_code_block_content(browser_navigate_result_text)
        config_dict = yaml_to_dict(yaml_config)
        page_info = extract_all_references_ordered(config_dict)
        save_file(page_info, f"{BASE_OUTPUT_DIR}/page_info.json")

        meta = {
            "url": start_url,
            "ref_count": len(page_info),
        }
        save_file(meta, f"{BASE_OUTPUT_DIR}/meta.json")

        return {
            "meta": meta,
            "page_info": page_info
        }

if __name__ == "__main__":
    query = "Top isekai anime today"

    asyncio.run(search_query(query))
