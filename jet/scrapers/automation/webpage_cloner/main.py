import asyncio
from pathlib import Path

from jet.scrapers.automation.grok_website_cloner import clone_after_render, generate_react_components


async def main():
    url = "http://example.com"
    output_dir = "output"

    # Clone webpage
    await clone_after_render(url, output_dir)

    # Generate React components
    html_path = Path(output_dir) / "index.html"
    html_content = html_path.read_text(encoding="utf-8")
    generate_react_components(html_content, output_dir)

if __name__ == "__main__":
    asyncio.run(main())
