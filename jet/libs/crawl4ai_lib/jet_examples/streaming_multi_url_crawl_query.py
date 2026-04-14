import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, List

import aiofiles
from jet.libs.crawl4ai_lib.async_web_crawler_manager import AsyncWebCrawlerManager

# ----------------------------------------------------------------------
# Application-specific logic (outside the class)
# ----------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSON = OUTPUT_DIR / "results.json"


async def init_json_file():
    if not RESULTS_JSON.exists():
        async with aiofiles.open(RESULTS_JSON, "w", encoding="utf-8") as f:
            await f.write("[]")


def calculate_relevance_score(result, user_query: str) -> float:
    if not getattr(result, "success", False):
        return 0.0

    markdown_obj = getattr(result, "markdown", None)
    if not markdown_obj:
        return 0.0

    fit_md = getattr(markdown_obj, "fit_markdown", None)
    raw_md = getattr(markdown_obj, "raw_markdown", None) or str(markdown_obj)

    content = fit_md or raw_md
    if not content or len(content) < 50:
        return 0.0

    ratio = len(content) / max(len(raw_md), 1)
    survival_score = min(1.0, ratio * 1.8)

    query_terms = [term.lower() for term in user_query.split() if len(term) > 2]
    if not query_terms:
        return round(survival_score, 3)

    text_lower = content.lower()
    hits = sum(text_lower.count(term) for term in query_terms)
    density = hits / max(len(content.split()), 1)

    keyword_bonus = min(0.6, density * 8)
    final_score = (survival_score * 0.65) + (keyword_bonus * 0.35)
    return round(min(1.0, final_score), 3)


async def save_result_to_json(result: Any, user_query: str) -> None:
    data: dict[str, Any] = {
        "url": getattr(result, "url", None),
        "success": getattr(result, "success", False),
        "timestamp": asyncio.get_event_loop().time(),
        "query": user_query,
    }

    if getattr(result, "success", False):
        data["title"] = (
            result.metadata.get("title") if getattr(result, "metadata", None) else None
        )
        data["status_code"] = getattr(result, "status_code", None)

        markdown_obj = getattr(result, "markdown", None)

        if hasattr(markdown_obj, "raw_markdown"):
            data["raw_markdown"] = markdown_obj.raw_markdown
            data["raw_markdown_length"] = len(markdown_obj.raw_markdown)

        fit_md = getattr(markdown_obj, "fit_markdown", None)
        if fit_md:
            data["fit_markdown"] = fit_md
            data["fit_markdown_length"] = len(fit_md)
            data["markdown"] = fit_md
        else:
            md_str = getattr(markdown_obj, "raw_markdown", None) or str(
                markdown_obj or ""
            )
            data["markdown"] = md_str
            data["markdown_length"] = len(md_str)

        data["relevance_score"] = calculate_relevance_score(result, user_query)

        extracted = getattr(result, "extracted_content", None)
        if extracted:
            data["extracted_content"] = extracted
            data["extracted_length"] = (
                len(extracted) if isinstance(extracted, str) else 0
            )
    else:
        data["error_message"] = getattr(result, "error_message", "Unknown error")
        data["status_code"] = getattr(result, "status_code", None)
        data["relevance_score"] = 0.0

    async with aiofiles.open(RESULTS_JSON, "r+", encoding="utf-8") as f:
        content = await f.read()
        results_list: List[dict] = json.loads(content) if content.strip() else []
        results_list.append(data)

        results_list.sort(
            key=lambda x: (x.get("relevance_score", 0.0), x.get("timestamp", 0)),
            reverse=True,
        )

        await f.seek(0)
        await f.truncate()
        await f.write(json.dumps(results_list, indent=2, ensure_ascii=False))


async def process_result(result: Any):
    """Print result and save to JSON. Uses the stored user_query from the manager."""
    # Access user_query from the manager instance
    user_query = getattr(crawler_manager, "_current_user_query", "")

    url = getattr(result, "url", "Unknown")

    if getattr(result, "success", False):
        title = (
            result.metadata.get("title") if getattr(result, "metadata", None) else "N/A"
        )
        markdown_obj = getattr(result, "markdown", None)

        raw_len = len(getattr(markdown_obj, "raw_markdown", "") or "")
        fit_len = len(getattr(markdown_obj, "fit_markdown", "") or "")
        score = calculate_relevance_score(result, user_query) if user_query else 0.0

        print(f"✅ [SUCCESS] {url}")
        print(f"   Title           : {title}")
        print(f"   Raw Markdown    : {raw_len:,} chars")
        if fit_len:
            print(f"   Fit Markdown    : {fit_len:,} chars (BM25 filtered)")
        print(f"   Relevance Score : {score:.3f} / 1.0")
        print("-" * 90)
    else:
        print(f"❌ [FAILED]  {url}")
        print(f"   Error : {getattr(result, 'error_message', 'Unknown error')}")
        if getattr(result, "status_code", None):
            print(f"   Status: {result.status_code}")
        print("-" * 90)

    await save_result_to_json(result, user_query)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

# Global reference so process_result can access the current user_query
crawler_manager: AsyncWebCrawlerManager | None = None


async def main():
    global crawler_manager

    await init_json_file()

    urls: List[str] = [
        "https://httpbin.org/html",
        "https://example.com",
        "https://www.python.org",
        "https://news.ycombinator.com",
        "https://github.com/unclecode/crawl4ai",
        "https://httpbin.org/json",
        "https://www.wikipedia.org",
    ]

    USER_QUERY = "AI web crawling and data extraction with Python"

    crawler_manager = AsyncWebCrawlerManager(
        headless=False,  # Change to True when you don't want visible tabs
        verbose=True,
        max_session_permit=10,
        semaphore_count=12,
        memory_threshold_percent=80.0,
        base_delay=(0.8, 2.0),
        delay_before_return_html=0.5,
        monitor_max_width=130,
    )

    await crawler_manager.crawl_many(
        urls=urls,
        process_result=process_result,  # Now only takes 'result'
        user_query=USER_QUERY,
        bm25_threshold=1.0,
    )

    # === Log results full path ===
    full_path = RESULTS_JSON.resolve().absolute()
    print("📁 Full results saved to:")
    print(f"   {full_path}")
    print("   (Open this file to see results sorted by relevance)")


if __name__ == "__main__":
    asyncio.run(main())
