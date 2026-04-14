#!/usr/bin/env python3
import argparse
import asyncio
import urllib.parse
from typing import List, Optional, TypedDict

from bs4 import BeautifulSoup
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    VirtualScrollConfig,
)


# ====================== TYPED DICT FOR VIDEO OBJECTS ======================
class Video(TypedDict):
    url: str
    text: str
    thumbnail: Optional[str]
    preview: Optional[str]


# ====================== DATA EXTRACTION HELPERS ======================
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


# ====================== MAIN CRAWL LOGIC ======================
async def main(
    query: str = "wife booty",
    headless: bool = False,
    scroll_count: int = 30,
    wait_after_scroll: float = 1.0,
    max_wait_for_growth: float = 8.0,  # Max seconds to wait for content to stabilize
    viewport_width: int = 1280,
    viewport_height: int = 720,
    browser_type: str = "chromium",
    wait_for_selector: str = ".text-secondary",
    zoom_level: float = 1.0,
):
    base_url = "https://missav.ws/en/search/"
    encoded_query = urllib.parse.quote(query)
    url = f"{base_url}{encoded_query}"

    print(f"🚀 Starting crawl for query: '{query}'")
    print(f"📍 URL: {url}")
    print(
        f"🔍 Zoom: {zoom_level * 100}% | Scrolls: {scroll_count} | Growth wait: {max_wait_for_growth}s"
    )

    browser_config = BrowserConfig(
        headless=headless,
        verbose=True,
        browser_type=browser_type,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )

    virtual_config = VirtualScrollConfig(
        container_selector="body",
        scroll_count=scroll_count,
        scroll_by="page_height",
        wait_after_scroll=wait_after_scroll,
    )

    # Fixed & clean JavaScript for waiting until content stops growing
    growth_wait_js = f"""
    (async () => {{
        const maxWaitMs = {int(max_wait_for_growth * 1000)};
        const checkInterval = 400;
        let lastHeight = document.body.scrollHeight;
        const startTime = Date.now();

        console.log(`Starting content growth wait (max {max_wait_for_growth}s)...`);

        while (Date.now() - startTime < maxWaitMs) {{
            await new Promise(r => setTimeout(r, checkInterval));
            
            const currentHeight = document.body.scrollHeight;
            if (currentHeight <= lastHeight) {{
                let elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                console.log(`✅ Content stabilized after ${{elapsed}}s (height: ${{currentHeight}})`);
                return true;
            }}
            lastHeight = currentHeight;
        }}

        let elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        console.log(`⏰ Max wait reached ({max_wait_for_growth}s, elapsed: ${{elapsed}}s), proceeding anyway`);
        return false;
    }})()
    """

    run_config = CrawlerRunConfig(
        virtual_scroll_config=virtual_config,
        wait_for=wait_for_selector,
        remove_overlay_elements=True,
        js_code=growth_wait_js,  # Runs after virtual scrolling completes
        delay_before_return_html=1.0,  # Extra safety buffer
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)

        if not result.success:
            print(f"❌ Crawl failed: {result.error_message}")
            return

        # Apply zoom if requested
        if zoom_level != 1.0 and hasattr(result, "page") and result.page:
            try:
                await result.page.evaluate(f"document.body.style.zoom = '{zoom_level}'")
                await asyncio.sleep(0.6)
                print(f"✅ Applied zoom: {zoom_level * 100}%")
            except Exception as e:
                print(f"⚠️ Could not apply zoom: {e}")

        videos = extract_data(result.html)

        print(f"✅ Successfully extracted {len(videos)} videos!")
        import json

        print(json.dumps(videos[:5], indent=2, ensure_ascii=False))


# ====================== CLI ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl MissAV search results with Crawl4AI + smart growth waiting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "query",
        nargs="?",
        default="wife booty",
        help="Search query",
    )
    parser.add_argument(
        "-H",
        "--headless",
        action="store_true",
        help="Run in headless mode",
    )
    parser.add_argument(
        "-s",
        "--scroll-count",
        type=int,
        default=30,
        help="Number of scroll operations",
    )
    parser.add_argument(
        "-w",
        "--wait-after-scroll",
        type=float,
        default=1.0,
        help="Fixed wait after each scroll (seconds)",
    )
    parser.add_argument(
        "-g",
        "--max-wait-growth",
        type=float,
        default=8.0,
        help="Maximum seconds to wait for content to stop growing before extraction",
    )
    parser.add_argument(
        "-z",
        "--zoom",
        type=float,
        default=0.8,
        help="Zoom level (0.8 = 80%%, 1.0 = 100%%)",
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
