"""
Minimal Format Scraping Example

This example demonstrates quick content extraction using
the minimal format for fast processing and overview.
"""

from swarms_tools.search.web_scraper import SuperFastScraper

from jet.file.utils import save_file
import os
import shutil

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Create scraper instance with optimized settings for speed
scraper = SuperFastScraper(
    timeout=8,
    max_workers=3,
    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.6998.205 Safari/537.36",
    strip_html=True,
    remove_scripts=True,
    remove_styles=True,
)

# URLs for quick content overview
quick_urls = [
    "https://news.ycombinator.com",
    "https://reddit.com/r/technology",
    "https://stackoverflow.com",
]

# Get minimal formatted results for rapid scanning
minimal_results = scraper.scrape_urls_formatted(
    urls=quick_urls,
    format_type="minimal",
    truncate=True,  # Truncate to 200 characters per result
)

# minimal_results contains compact summaries like:
# "Page Title: First 200 characters of content..."
# Perfect for quick scanning and overview

compact_summaries = minimal_results

save_file(compact_summaries, f"{OUTPUT_DIR}/compact_summaries.md")

# Ideal for:
# - Content previews
# - Quick scanning
# - Feed generation
# - Summary dashboards
