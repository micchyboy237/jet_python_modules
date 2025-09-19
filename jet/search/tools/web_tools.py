from typing import List

from fake_useragent import UserAgent

from jet.search.web_scraper import SuperFastScraper


def scrape_urls(
    urls: List[str],
    format_type: str = "detailed",
    truncate: bool = True
) -> str:
    """
    Scrape content from a list of URLs and format it into different string formats.

    Args:
        urls: List of URLs to scrape
        format_type: Format type ('detailed', 'summary', 'minimal', 'markdown', 'full')
        truncate: Whether to truncate text in summary/minimal formats (default: True)

    Returns:
        Formatted string containing scraped content
    """
    ua = UserAgent()
    scraper = SuperFastScraper(
        timeout=8,
        max_workers=3,
        user_agent=ua.random,
        strip_html=True,
        remove_scripts=True,
        remove_styles=True,
    )

    formatted_result = scraper.scrape_urls_formatted(
        urls=urls,
        format_type=format_type,
        truncate=True,
    )

    return formatted_result
