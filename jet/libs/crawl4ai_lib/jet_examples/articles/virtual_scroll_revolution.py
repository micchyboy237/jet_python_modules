# virtual_scroll_revolution.py
# Updated with proper wait timeouts to prevent "Container not found" or navigation timeouts

import asyncio
import json
import shutil
from pathlib import Path

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    LLMExtractionStrategy,
    VirtualScrollConfig,
)
from jet.libs.crawl4ai_lib.adaptive_config import get_llm_config

# Setup output directory
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_demo_output_dir(demo_number: int, demo_name: str) -> Path:
    """Create numbered subdirectory: 01_basic_virtual_scroll, etc."""
    dir_name = f"{demo_number:02d}_{demo_name}"
    demo_dir = OUTPUT_DIR / dir_name
    demo_dir.mkdir(parents=True, exist_ok=True)
    return demo_dir


async def demo_basic_virtual_scroll():
    """Demo 1: Basic virtual scroll configuration"""
    demo_dir = get_demo_output_dir(1, "basic_virtual_scroll")

    virtual_config = VirtualScrollConfig(
        container_selector="div[role='feed']",
        scroll_count=25,
        scroll_by="container_height",
        wait_after_scroll=0.8,  # Increased slightly for stability
    )

    config = CrawlerRunConfig(
        virtual_scroll_config=virtual_config,
        page_timeout=120000,  # 2 minutes timeout
        wait_until="domcontentloaded",
        delay_before_return_html=1.0,
        verbose=True,
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url="https://x.com/explore", config=config)

        (demo_dir / "captured.html").write_text(result.html, encoding="utf-8")
        (demo_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "demo": "1 - Basic Virtual Scroll",
                    "url": result.url,
                    "html_length": len(result.html),
                    "status_code": result.status_code,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"Captured {len(result.html)} characters of HTML")
        print(f"✅ Results saved to: {demo_dir}")


async def demo_twitter_thread_capture():
    """Demo 2: Capturing a full Twitter/X thread"""
    demo_dir = get_demo_output_dir(2, "twitter_thread_capture")

    virtual_config = VirtualScrollConfig(
        container_selector="[data-testid='primaryColumn']",
        scroll_count=35,
        scroll_by="container_height",
        wait_after_scroll=1.5,  # Twitter often needs more time
    )

    config = CrawlerRunConfig(
        virtual_scroll_config=virtual_config,
        extraction_strategy=LLMExtractionStrategy(
            llm_config=get_llm_config(strategy="llm"),
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "author": {"type": "string"},
                        "content": {"type": "string"},
                        "timestamp": {"type": "string"},
                        "replies": {"type": "integer"},
                        "retweets": {"type": "integer"},
                        "likes": {"type": "integer"},
                    },
                },
            },
        ),
        page_timeout=180000,  # 3 minutes for threads + LLM extraction
        wait_until="domcontentloaded",
        delay_before_return_html=2.0,
        verbose=True,
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://x.com/elonmusk",
            config=config,
        )

        (demo_dir / "thread.html").write_text(result.html, encoding="utf-8")

        try:
            tweets = json.loads(result.extracted_content or "[]")
            (demo_dir / "tweets.json").write_text(
                json.dumps(tweets, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"Captured {len(tweets)} tweets")
        except Exception as e:
            print(f"Note: Could not parse extracted_content ({e})")

        print(f"✅ Results saved to: {demo_dir}")


async def demo_mixed_content_handling():
    """Demo 3: Mixed content handling"""
    demo_dir = get_demo_output_dir(3, "mixed_content")

    virtual_config = VirtualScrollConfig(
        container_selector="#hnmain table",
        scroll_count=20,
        scroll_by="container_height",
        wait_after_scroll=0.6,
    )

    config = CrawlerRunConfig(
        virtual_scroll_config=virtual_config,
        page_timeout=90000,  # 90 seconds
        wait_until="domcontentloaded",
        delay_before_return_html=1.0,
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url="https://news.ycombinator.com/", config=config)

        (demo_dir / "feed.html").write_text(result.html, encoding="utf-8")

        print(f"Total HTML length: {len(result.html)} characters")
        print(f"✅ Results saved to: {demo_dir}")


async def demo_fast_vs_careful_scrolling():
    """Demo 4: Fast scrolling configuration"""
    demo_dir = get_demo_output_dir(4, "fast_scrolling")

    fast_config = VirtualScrollConfig(
        container_selector="div[role='feed'], #hnmain",
        scroll_count=30,
        scroll_by=700,
        wait_after_scroll=0.4,
    )

    config = CrawlerRunConfig(
        virtual_scroll_config=fast_config,
        page_timeout=120000,
        wait_until="domcontentloaded",
        delay_before_return_html=1.0,
        verbose=True,
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url="https://x.com/explore", config=config)

        (demo_dir / "fast_scroll_result.html").write_text(result.html, encoding="utf-8")

        print(f"Fast scroll completed - captured {len(result.html)} characters")
        print(f"✅ Results saved to: {demo_dir}")


async def demo_debug_with_visible_browser():
    """Demo 5: Debug with visible browser"""
    demo_dir = get_demo_output_dir(5, "debug_visible")

    browser_config = BrowserConfig(headless=False)

    virtual_config = VirtualScrollConfig(
        container_selector="div[role='feed']",
        scroll_count=12,
        scroll_by="container_height",
        wait_after_scroll=1.2,
    )

    config = CrawlerRunConfig(
        virtual_scroll_config=virtual_config,
        page_timeout=120000,
        wait_until="domcontentloaded",
        delay_before_return_html=1.5,
        verbose=True,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url="https://x.com/explore", config=config)

        (demo_dir / "debug_capture.html").write_text(result.html, encoding="utf-8")

        print("Debug session completed – browser was visible")
        print(f"✅ Results saved to: {demo_dir}")


async def demo_complete_starter_template():
    """Demo 6: Complete ready-to-use starter template"""
    demo_dir = get_demo_output_dir(6, "starter_template")

    virtual_config = VirtualScrollConfig(
        container_selector="#hnmain",
        scroll_count=25,
        scroll_by="container_height",
        wait_after_scroll=0.8,
    )

    config = CrawlerRunConfig(
        virtual_scroll_config=virtual_config,
        page_timeout=120000,
        wait_until="domcontentloaded",
        delay_before_return_html=1.5,
        verbose=True,
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://news.ycombinator.com/",
            config=config,
        )

        (demo_dir / "complete_capture.html").write_text(result.html, encoding="utf-8")

        print(f"Captured {len(result.html)} characters of content")
        print(f"✅ Results saved to: {demo_dir}")


# ========================
# Helper
# ========================


async def run_demo(demo_name: str):
    demos = {
        "basic": demo_basic_virtual_scroll,
        "twitter": demo_twitter_thread_capture,
        "mixed": demo_mixed_content_handling,
        "fast": demo_fast_vs_careful_scrolling,
        "debug": demo_debug_with_visible_browser,
        "starter": demo_complete_starter_template,
    }

    if demo_name in demos:
        print(f"\n{'=' * 75}")
        print(f"🚀 Running Demo: {demo_name}")
        print(f"{'=' * 75}\n")
        await demos[demo_name]()
    else:
        print("Available demos:", list(demos.keys()))


if __name__ == "__main__":
    # Run all demos
    for name in ["basic", "twitter", "mixed", "fast", "debug", "starter"]:
        asyncio.run(run_demo(name))
