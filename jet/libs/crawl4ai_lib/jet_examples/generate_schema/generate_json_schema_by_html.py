import asyncio
import json
import os
import shutil
from pathlib import Path

from crawl4ai import (
    AsyncWebCrawler,
    CacheMode,
    CrawlerRunConfig,
    JsonCssExtractionStrategy,
)
from jet.adapters.llama_cpp.tokens import count_tokens
from jet.file.utils import load_file, save_file
from jet.libs.crawl4ai_lib.adaptive_config import get_llm_config
from jet.libs.crawl4ai_lib.preprocessors import preprocess_for_schema_generation

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# ====================== CLEAN HTML FOR SCHEMA GENERATION ======================
html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/playwright/generated/run_scrape_urls_playwright/missav_ws_dm223_en/sync_results/page.html"
html = load_file(html_file)


# def clean_html_for_schema(raw_html: str) -> str:
#     """Strip scripts, styles, nav, headers, footers, etc. to drastically reduce token count
#     while preserving the DOM structure needed for CSS selector generation."""
#     soup = BeautifulSoup(raw_html, "html.parser")

#     # Remove non-content elements
#     for tag in soup.find_all(["script", "style", "noscript", "svg", "path", "iframe"]):
#         tag.decompose()

#     # Remove comments
#     for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
#         comment.extract()

#     # Keep only body (or whole soup if no body)
#     body = soup.find("body") or soup

#     # Remove common non-main containers
#     for selector in [
#         "nav",
#         "header",
#         "footer",
#         "aside",
#         ".sidebar",
#         "#sidebar",
#         ".ad",
#         ".ads",
#         ".cookie",
#     ]:
#         for elem in body.select(selector):
#             elem.decompose()

#     return str(body)


cleaned_html = preprocess_for_schema_generation(html)

model = os.getenv("LLAMA_CPP_LLM_MODEL")

orig_tokens = count_tokens(html, model)
cleaned_tokens = count_tokens(cleaned_html, model)

print(f"Original HTML tokens (approx): ~{orig_tokens}")
print(f"Cleaned HTML tokens (approx):  ~{cleaned_tokens}")

save_file(cleaned_html, OUTPUT_DIR / "page.html")

# ====================== GENERATE SCHEMA (now safe) ======================
# Option A: Use a high-context model (recommended – Gemini 1M context is perfect)
schema = JsonCssExtractionStrategy.generate_schema(
    cleaned_html,
    llm_config=get_llm_config(strategy="llm"),
    # Optional: describe what you want for better selectors
    # query="Extract all video titles, links, thumbnails, durations, and view counts from the page.",
)

# Option B: Keep your adaptive config (if you updated it to a large-context model)
# schema = JsonCssExtractionStrategy.generate_schema(
#     cleaned_html, llm_config=get_llm_config(strategy="llm")
# )

save_file(schema, OUTPUT_DIR / "schema.json")

# ====================== FAST EXTRACTION (unchanged, LLM-free) ======================
strategy = JsonCssExtractionStrategy(schema)


async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="raw://"
            + html,  # use original raw HTML here (extraction is fast & selector-based)
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                extraction_strategy=strategy,
            ),
        )
        data = json.loads(result.extracted_content)
        print("Extracted content:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        save_file(data, OUTPUT_DIR / "extracted_content.json")


if __name__ == "__main__":
    asyncio.run(main())
