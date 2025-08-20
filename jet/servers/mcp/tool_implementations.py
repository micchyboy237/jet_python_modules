import asyncio
from urllib.parse import urlparse, urljoin
from typing import AsyncIterator
from playwright.async_api import async_playwright
from mcp.server.fastmcp.server import Context
from jet.servers.mcp.models import FileInput, FileOutput, UrlInput, UrlOutput, SummarizeTextInput, SummarizeTextOutput

PLAYWRIGHT_CHROMIUM_EXECUTABLE = "/Users/jethroestrada/Library/Caches/ms-playwright/chromium-1181/chrome-mac/Chromium.app/Contents/MacOS/Chromium"


async def read_file(arguments: FileInput, ctx: Context) -> FileOutput:
    await ctx.info(f"Reading file: {arguments.file_path}")
    try:
        with open(arguments.file_path, "r", encoding=arguments.encoding) as f:
            content = f.read()
        await ctx.report_progress(100, 100, "File read successfully")
        return FileOutput(content=content)
    except Exception as e:
        await ctx.error(f"Error reading file: {str(e)}")
        return FileOutput(content=f"Error reading file: {str(e)}")


async def navigate_to_url(arguments: UrlInput, ctx: Context) -> UrlOutput:
    await ctx.info(f"Navigating to {arguments.url}")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=False,
                executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE,
            )
            page = await browser.new_page()
            await page.goto(arguments.url)
            title = await page.title()
            link_elements = await page.query_selector_all('a[href]')
            links = []
            parsed_url = urlparse(arguments.url)
            base_domain = parsed_url.netloc
            for element in link_elements:
                href = await element.get_attribute('href')
                if href:
                    absolute_url = urljoin(arguments.url, href)
                    parsed_link = urlparse(absolute_url)
                    if not parsed_link.netloc or parsed_link.netloc == base_domain:
                        links.append(absolute_url)
            text_content = await page.evaluate('''() => {
                return document.body.innerText.trim();
            }''')
            await browser.close()
        await ctx.report_progress(100, 100, "Navigation complete")
        return UrlOutput(
            title=f"Navigated to {arguments.url}. Page title: {title}",
            nav_links=links or None,
            text_content=text_content or None
        )
    except Exception as e:
        await ctx.error(f"Error navigating to {arguments.url}: {str(e)}")
        return UrlOutput(title=f"Error navigating to {arguments.url}: {str(e)}", nav_links=None, text_content=None)


async def summarize_text(arguments: SummarizeTextInput, ctx: Context) -> SummarizeTextOutput:
    await ctx.info(f"Summarizing text (max {arguments.max_words} words)")
    try:
        words = arguments.text.split()
        summary_words = words[:arguments.max_words]
        summary = " ".join(summary_words)
        if len(words) > arguments.max_words:
            summary += "..."
        word_count = len(summary_words)
        await ctx.report_progress(100, 100, "Summary generated")
        return SummarizeTextOutput(summary=summary, word_count=word_count)
    except Exception as e:
        await ctx.error(f"Error summarizing text: {str(e)}")
        return SummarizeTextOutput(summary=f"Error: {str(e)}", word_count=0)


async def process_data(data: str, ctx: Context) -> str:
    for i in range(1, 101, 10):
        await ctx.report_progress(i, 100, f"Processing step {i}%")
        await asyncio.sleep(0.1)
    return f"Processed: {data}"
