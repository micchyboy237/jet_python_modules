import argparse
import asyncio
import urllib.parse
from typing import List, Optional, TypedDict

from bs4 import BeautifulSoup
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    UndetectedAdapter,
    VirtualScrollConfig,
)
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy


class Video(TypedDict):
    url: str
    text: str
    thumbnail: Optional[str]
    preview: Optional[str]


def get_src_or_data_src(tag) -> Optional[str]:
    if not tag:
        return None
    src = tag.get("src") or tag.get("data-src")
    return src.strip() if src else None


def extract_data(html: str) -> List[Video]:
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.select(".text-secondary")
    data: List[Video] = []
    for a in anchors:
        url = a.get("href", "").strip()
        text = a.get_text(strip=True)
        if "#" in url:
            url = url.split("#")[0]
        if not url or not text:
            continue
        container = a
        while container and container.name != "body":
            if container.find_all("img") and container.find_all("video"):
                break
            container = container.parent
        else:
            container = None
        if not container:
            data.append({"url": url, "text": text, "thumbnail": None, "preview": None})
            continue
        img = container.find("img")
        thumbnail = get_src_or_data_src(img)
        video = container.find("video")
        preview = None
        if video:
            preview = get_src_or_data_src(video)
            if not preview:
                source = video.find("source")
                preview = get_src_or_data_src(source)
        data.append(
            {
                "url": url,
                "text": text,
                "thumbnail": thumbnail,
                "preview": preview,
            }
        )
    return data


async def main(
    query: str = "wife strips tease",
    headless: bool = False,
    scroll_count: int = 30,
    wait_after_scroll: float = 5.0,
    max_wait_for_growth: float = 12.0,
    viewport_width: int = 1280,
    viewport_height: int = 720,
    browser_type: str = "chromium",
    wait_for_selector: str = ".text-secondary",
    zoom_level: float = 1.0,
):
    # Smart default: 80% of viewport height (exactly what the print message promised)
    scroll_by_pixels = int(viewport_height * 0.8)

    base_url = "https://missav.ws/en/search/"
    encoded_query = urllib.parse.quote(query)
    url = f"{base_url}{encoded_query}"
    print(f"🚀 Starting crawl for query: '{query}'")
    print(f"📍 URL: {url}")
    print(
        f"🔍 Zoom: {zoom_level * 100}% | Scroll by: {scroll_by_pixels}px each step | Scrolls: {scroll_count} | "
        f"Growth wait: {max_wait_for_growth}s | Anti-bot: Undetected + Stealth"
    )

    browser_config = BrowserConfig(
        headless=headless,
        verbose=True,
        browser_type=browser_type,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        enable_stealth=True,
    )
    adapter = UndetectedAdapter()
    strategy = AsyncPlaywrightCrawlerStrategy(
        browser_config=browser_config,
        browser_adapter=adapter,
    )

    virtual_config = VirtualScrollConfig(
        container_selector="body",  # updated name (was container_selector)
        scroll_by=scroll_by_pixels,  # now fully respected!
        scroll_count=scroll_count,  # updated name (was scroll_count)
        wait_after_scroll=wait_after_scroll,  # updated name (was wait_after_scroll)
    )

    growth_wait_js = f"""
    (async () => {{
        const MAX_WAIT_MS = {int(max_wait_for_growth * 1000)};
        const CHECK_INTERVAL = 400;
        const EXTRA_BOTTOM_CYCLES = 4; // Try up to 4 extra scroll/checks at bottom
        const BOTTOM_DELAY_MS = 2500; // 2.5s human-like pause at bottom
        let lastHeight = document.body.scrollHeight;
        const startTime = Date.now();
        console.log(`🚀 Starting advanced content stabilization + bottom re-checks (max {max_wait_for_growth}s)...`);
        // Phase 1: Standard growth stabilization (after VirtualScroll finishes)
        while (Date.now() - startTime < MAX_WAIT_MS) {{
            await new Promise(r => setTimeout(r, CHECK_INTERVAL));
            const currentHeight = document.body.scrollHeight;
            if (currentHeight <= lastHeight) {{
                console.log(`✅ Initial stabilization after ${{((Date.now() - startTime)/1000).toFixed(1)}}s`);
                break;
            }}
            lastHeight = currentHeight;
        }}
        // Phase 2: Extra bottom checks - this fixes quick session end
        let extraAttempts = 0;
        while (extraAttempts < EXTRA_BOTTOM_CYCLES && (Date.now() - startTime) < MAX_WAIT_MS) {{
            await new Promise(r => setTimeout(r, BOTTOM_DELAY_MS)); // delay at bottom
            const currentHeight = document.body.scrollHeight;
            if (currentHeight > lastHeight) {{
                const added = currentHeight - lastHeight;
                console.log(`📈 New content loaded (+${{added}}px) — scrolling again...`);
                window.scrollBy(0, window.innerHeight * 0.8); // same 80% scroll
                await new Promise(r => setTimeout(r, {int(wait_after_scroll * 1000)})); // wait for lazy load
                lastHeight = document.body.scrollHeight;
                extraAttempts++;
            }} else {{
                console.log(`✅ No more content after bottom delay (attempt ${{extraAttempts + 1}})`);
                break;
            }}
        }}
        let elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        console.log(`🏁 Final stabilization complete after ${{elapsed}}s (extra bottom cycles: ${{extraAttempts}})`);
        return true;
    }})()
    """

    zoom_js = None
    if zoom_level != 1.0:
        zoom_js = f"""
            document.body.style.zoom = '{zoom_level}';
            console.log('✅ Zoom applied early: {zoom_level * 100}%');
        """

    run_config = CrawlerRunConfig(
        virtual_scroll_config=virtual_config,
        wait_for=wait_for_selector,
        remove_overlay_elements=True,
        js_code_before_wait=zoom_js,
        js_code=growth_wait_js,
        delay_before_return_html=2.0,
    )

    async with AsyncWebCrawler(
        crawler_strategy=strategy, config=browser_config
    ) as crawler:
        result = await crawler.arun(url=url, config=run_config)
        if not result.success:
            print(f"❌ Crawl failed: {result.error_message}")
            return
        videos = extract_data(result.html)
        print(f"✅ Successfully extracted {len(videos)} videos!")
        import json

        print(json.dumps(videos[:5], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl MissAV search results with Crawl4AI + latest anti-bot + smart bottom re-scroll",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="wife strips tease",
        help="Search query",
    )
    parser.add_argument(
        "-H",
        "--headless",
        action="store_true",
        help="Run in headless mode (recommended: False for best anti-bot)",
    )
    parser.add_argument(
        "-s",
        "--scroll-count",
        type=int,
        default=30,
        help="Number of scroll operations",
    )
    parser.add_argument(
        "--scroll-by-pixels",
        type=int,
        default=None,
        help="Pixels to scroll each step (default = 80%% of viewport height - auto-calculated)",
    )
    parser.add_argument(
        "-w",
        "--wait-after-scroll",
        type=float,
        default=5.0,
        help="Fixed wait after each scroll (seconds) — increased for lazy-loading",
    )
    parser.add_argument(
        "-g",
        "--max-wait-growth",
        type=float,
        default=12.0,
        help="Maximum seconds to wait for content to stop growing + extra bottom checks",
    )
    parser.add_argument(
        "-z",
        "--zoom",
        type=float,
        default=0.8,
        help="Zoom level (0.8 = 80%%, 1.0 = 100%%) — applied early",
    )
    parser.add_argument(
        "--viewport-width",
        type=int,
        default=1280,
        help="Viewport width",
    )
    parser.add_argument(
        "--viewport-height",
        type=int,
        default=720,
        help="Viewport height",
    )
    parser.add_argument(
        "--browser-type",
        type=str,
        choices=["chromium", "firefox", "webkit"],
        default="chromium",
        help="Browser engine",
    )
    parser.add_argument(
        "--wait-for",
        type=str,
        default=".text-secondary",
        help="CSS selector to wait for",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            query=args.query,
            headless=args.headless,
            scroll_count=args.scroll_count,
            wait_after_scroll=args.wait_after_scroll,
            max_wait_for_growth=args.max_wait_growth,
            viewport_width=args.viewport_width,
            viewport_height=args.viewport_height,
            browser_type=args.browser_type,
            wait_for_selector=args.wait_for,
            zoom_level=args.zoom,
        )
    )
