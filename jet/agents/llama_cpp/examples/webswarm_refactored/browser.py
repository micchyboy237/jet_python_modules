import logging

import trafilatura
from playwright.async_api import async_playwright

from .config import DOC_CHAR_LIMIT

logger = logging.getLogger("webswarm")


class BrowserManager:
    _ctx = None

    @classmethod
    async def get_context(cls):
        if cls._ctx is None:
            pw = await async_playwright().start()
            browser = await pw.chromium.launch(headless=True)
            cls._ctx = await browser.new_context(
                user_agent="Mozilla/5.0 (ResearchBot/1.0)",
                viewport={"width": 1280, "height": 800},
            )
        return cls._ctx


async def extract_page(url: str) -> dict:
    ctx = await BrowserManager.get_context()
    page = await ctx.new_page()
    try:
        await page.goto(url, timeout=15000, wait_until="domcontentloaded")
        html = await page.content()
        text = trafilatura.extract(html, include_comments=False) or ""
        return {"url": url, "text": text[:DOC_CHAR_LIMIT]}
    except Exception as e:
        logger.warning(f"Browser fail {url}: {e}")
        return {"url": url, "text": "", "error": str(e)}
    finally:
        await page.close()
