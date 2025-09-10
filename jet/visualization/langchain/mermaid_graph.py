import os
from pathlib import Path
import asyncio
import pyppeteer
from langchain_core.runnables.graph import MermaidDrawMethod
from jet.logger import logger
from jet.scrapers.browser.config import PLAYWRIGHT_CHROMIUM_EXECUTABLE


async def safe_launch_browser(max_retries: int = 5, retry_delay: float = 2.0):
    """Safely launch a browser with retries for Mac M1 compatibility."""
    for attempt in range(max_retries):
        try:
            browser = await pyppeteer.launch(
                executablePath=PLAYWRIGHT_CHROMIUM_EXECUTABLE,
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
            logger.info("Browser launched successfully")
            return browser
        except Exception as e:
            logger.warning(f"Browser launch attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                logger.error(
                    f"Failed to launch browser after {max_retries} attempts")
                raise
    raise Exception("Failed to launch browser")


def render_mermaid_graph(
    agent,
    output_filename="graph_output.png",
    draw_method=MermaidDrawMethod.PYPPETEER,
    max_retries: int = 5,
    retry_delay: float = 2.0,
    open_file: bool = False,
    **kwargs
):
    """
    Generates a Mermaid graph PNG from the agent and optionally opens it using the system's default viewer on macOS.

    Args:
        agent: An object with a get_graph().draw_mermaid_png() method.
        output_filename (str): Name of the file to save the PNG to.
        draw_method: Drawing method (e.g., MermaidDrawMethod.PYPPETEER).
        max_retries: Maximum number of browser launch retries.
        retry_delay: Delay between retries in seconds.
        open_file: Whether to open the file with the default viewer.
        **kwargs: Additional arguments for get_graph().
    """
    try:
        # Ensure event loop is available and not closed
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def render():
            browser = await safe_launch_browser(max_retries, retry_delay)
            try:
                png_bytes = await agent.get_graph(**kwargs).draw_mermaid_png(
                    draw_method=draw_method, browser=browser
                )
                output_path = Path(output_filename)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(png_bytes)
                logger.info(f"Graph rendered successfully to {output_path}")
                if open_file:
                    os.system(f"open {output_path}")
            finally:
                await browser.close()
                logger.debug("Browser closed")

        loop.run_until_complete(render())
    except Exception as e:
        logger.error(f"Error rendering Mermaid graph: {e}")
        raise
