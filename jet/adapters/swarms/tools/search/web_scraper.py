"""
Super Fast Web Scraper for extracting information from URLs into formatted plain strings.

This module provides ultra-high-performance web scraping capabilities using:
- httpx: Fast HTTP client with HTTP/2 support
- lxml.html: Fastest HTML parsing available (2-3x faster than BeautifulSoup)
- ThreadPoolExecutor: True concurrent processing for multiple URLs
- XPath: Ultra-fast element selection and text extraction
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List
from urllib.parse import urljoin

import httpx
from lxml import html
from lxml.html import HtmlElement

from jet.logger import logger


@dataclass
class ScrapedContent:
    """Container for scraped content with metadata."""

    url: str
    title: str
    text: str
    links: List[str]
    images: List[str]
    timestamp: float
    status_code: int
    content_type: str
    word_count: int


class SuperFastScraper:
    """
    Ultra-fast web scraper with ThreadPoolExecutor for concurrent processing.
    This is the main scraper class that provides all functionality.
    """

    def __init__(
        self,
        timeout: int = 10,
        max_retries: int = 3,
        user_agent: str = "Mozilla/5.0 (compatible; SuperFastScraper/1.0)",
        max_workers: int = 10,
        strip_html: bool = True,
        remove_scripts: bool = True,
        remove_styles: bool = True,
        remove_comments: bool = True,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_headers = {"User-Agent": user_agent}
        self.max_workers = max_workers
        self.strip_html = strip_html
        self.remove_scripts = remove_scripts
        self.remove_styles = remove_styles
        self.remove_comments = remove_comments

    def scrape_single_url(self, url: str) -> ScrapedContent:
        """Scrape a single URL synchronously."""
        start_time = time.time()
        logger.debug(f"Starting scrape for URL: {url}")

        try:
            # Use httpx in sync mode for better performance
            with httpx.Client(
                timeout=self.timeout,
                headers=self.user_headers,
                follow_redirects=True,
                http2=True,
            ) as client:
                logger.debug(f"Sending HTTP request to {url}")
                response = client.get(url)
                response.raise_for_status()
                logger.info(f"Successfully fetched {url} with status code {response.status_code}")

                # Parse HTML with lxml.html for maximum speed
                soup = html.fromstring(response.content)
                logger.debug(f"Parsed HTML content for {url}")

                # Extract content
                title = self._extract_title(soup)
                logger.debug(f"Extracted title: {title[:50]}..." if len(title) > 50 else f"Extracted title: {title}")
                text = self._extract_text(soup)
                logger.debug(f"Extracted text (first 50 chars): {text[:50]}..." if len(text) > 50 else f"Extracted text: {text}")
                links = self._extract_links(soup, url)
                logger.debug(f"Extracted {len(links)} links from {url}")
                images = self._extract_images(soup, url)
                logger.debug(f"Extracted {len(images)} images from {url}")

                result = ScrapedContent(
                    url=url,
                    title=title,
                    text=text,
                    links=links,
                    images=images,
                    timestamp=start_time,
                    status_code=response.status_code,
                    content_type=response.headers.get("content-type", ""),
                    word_count=len(text.split()),
                )
                logger.info(f"Completed scraping {url} in {time.time() - start_time:.2f} seconds")
                return result

        except Exception as e:
            logger.error(f"Failed to scrape {url}: {str(e)}")
            # Return error content
            return ScrapedContent(
                url=url,
                title="Error",
                text=f"Failed to scrape: {str(e)}",
                links=[],
                images=[],
                timestamp=start_time,
                status_code=0,
                content_type="",
                word_count=0,
            )

    def scrape_urls(self, urls: List[str]) -> List[ScrapedContent]:
        """
        Scrape multiple URLs concurrently using ThreadPoolExecutor.

        Args:
            urls: List of URLs to scrape

        Returns:
            List of ScrapedContent objects
        """
        logger.info(f"Starting concurrent scrape for {len(urls)} URLs")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all scraping tasks
            future_to_url = {executor.submit(self.scrape_single_url, url): url for url in urls}
            logger.debug(f"Submitted {len(future_to_url)} scraping tasks")

            results = []
            # Collect results as they complete
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.debug(f"Completed scraping task for {url}")
                except Exception as e:
                    logger.error(f"Scraping task failed for {url}: {str(e)}")
                    # Create error content for failed scrapes
                    results.append(
                        ScrapedContent(
                            url=url,
                            title="Error",
                            text=f"Scraping failed: {str(e)}",
                            links=[],
                            images=[],
                            timestamp=time.time(),
                            status_code=0,
                            content_type="",
                            word_count=0,
                        )
                    )

        # Sort results to maintain original URL order
        url_to_result = {result.url: result for result in results}
        ordered_results = [url_to_result.get(url, url_to_result.get("Error")) for url in urls if url in url_to_result]
        logger.info(f"Completed scraping {len(ordered_results)} URLs")
        return ordered_results

    def scrape_urls_formatted(
        self,
        urls: List[str],
        format_type: str = "detailed",
        truncate: bool = True,
    ) -> str:
        """
        Scrape multiple URLs and return formatted string.

        Args:
            urls: List of URLs to scrape
            format_type: Format type ('detailed', 'summary', 'minimal', 'markdown', 'full')
            truncate: Whether to truncate text in summary/minimal formats

        Returns:
            Formatted string containing all scraped results
        """
        results = self.scrape_urls(urls)

        if not results:
            return "No URLs were successfully scraped."

        # Format each result
        formatted_results = []
        for i, content in enumerate(results, 1):
            if content.title == "Error":
                formatted_results.append(
                    f"URL {i}: {content.url}\nStatus: {content.title}\nError: {content.text}\n"
                )
            else:
                formatted_results.append(
                    f"URL {i}: {content.url}\n{self.format_content(content, format_type, truncate)}\n"
                )

        # Join all results with separators
        separator = "\n" + "=" * 80 + "\n"
        return separator.join(formatted_results)

    def _extract_title(self, soup: HtmlElement) -> str:
        """Extract page title."""
        logger.debug("Extracting page title")
        title_tag = soup.find(".//title")
        if title_tag is not None:
            title = self._clean_text(title_tag.text_content())
            logger.debug(f"Found title tag: {title[:50]}..." if len(title) > 50 else f"Found title tag: {title}")
            return title

        # Fallback to h1 or meta title
        h1 = soup.find(".//h1")
        if h1 is not None:
            title = self._clean_text(h1.text_content())
            logger.debug(f"Found h1 title: {title[:50]}..." if len(title) > 50 else f"Found h1 title: {title}")
            return title

        meta_title = soup.find(".//meta[@name='title']")
        if meta_title is not None:
            title = self._clean_text(meta_title.get("content", ""))
            logger.debug(f"Found meta title: {title[:50]}..." if len(title) > 50 else f"Found meta title: {title}")
            return title

        logger.warning("No title found in document")
        return "No title found"

    def _extract_text(self, soup: HtmlElement) -> str:
        """Extract clean, readable text content."""
        logger.debug("Extracting text content")
        # Remove unwanted elements
        if self.remove_scripts:
            scripts = soup.xpath(".//script | .//noscript")
            logger.debug(f"Removing {len(scripts)} script/noscript elements")
            for script in scripts:
                script.getparent().remove(script)

        if self.remove_styles:
            styles = soup.xpath(".//style")
            logger.debug(f"Removing {len(styles)} style elements")
            for style in styles:
                style.getparent().remove(style)

        if self.remove_comments:
            comments = soup.xpath(".//comment()")
            logger.debug(f"Removing {len(comments)} comment elements")
            for comment in comments:
                comment.getparent().remove(comment)

        # Get text from body or main content areas
        content_selectors = [
            ".//main",
            ".//article",
            ".//*[contains(@class, 'content')]",
            ".//*[contains(@class, 'post-content')]",
            ".//*[contains(@class, 'entry-content')]",
            ".//*[contains(@class, 'article-content')]",
            ".//*[contains(@class, 'story-content')]",
            ".//*[contains(@class, 'text-content')]",
        ]

        content = ""
        for selector in content_selectors:
            elements = soup.xpath(selector)
            if elements:
                content = " ".join([elem.text_content() for elem in elements])
                logger.debug(f"Extracted content from selector: {selector}")
                break

        # Fallback to body if no specific content area found
        if not content:
            body = soup.xpath(".//body")
            if body:
                content = body[0].text_content()
                logger.debug("Extracted content from body")
            else:
                content = soup.text_content()
                logger.debug("Extracted content from entire document")

        cleaned_content = self._clean_text(content)
        logger.debug(f"Cleaned text content (first 50 chars): {cleaned_content[:50]}..." if len(cleaned_content) > 50 else f"Cleaned text content: {cleaned_content}")
        return cleaned_content

    def _extract_links(self, soup: HtmlElement, base_url: str) -> List[str]:
        """Extract all links from the page."""
        logger.debug(f"Extracting links from {base_url}")
        links = []
        for link in soup.xpath(".//a[@href]"):
            href = link.get("href")
            absolute_url = urljoin(base_url, href)
            if absolute_url.startswith(("http://", "https://")):
                links.append(absolute_url)
        unique_links = list(set(links))  # Remove duplicates
        logger.debug(f"Extracted {len(unique_links)} unique links")
        return unique_links

    def _extract_images(self, soup: HtmlElement, base_url: str) -> List[str]:
        """Extract all image URLs from the page."""
        logger.debug(f"Extracting images from {base_url}")
        images = []
        for img in soup.xpath(".//img[@src]"):
            src = img.get("src")
            absolute_url = urljoin(base_url, src)
            if absolute_url.startswith(("http://", "https://")):
                images.append(absolute_url)
        unique_images = list(set(images))  # Remove duplicates
        logger.debug(f"Extracted {len(unique_images)} unique images")
        return unique_images

    def _clean_text(self, text: str) -> str:
        """Clean and format text content."""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove special characters but keep readable ones
        text = re.sub(r"[^\w\s\-.,!?;:()]", "", text)

        # Clean up punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"([.,!?;:])\s*([.,!?;:])", r"\1", text)

        return text.strip()

    def format_content(
        self,
        content: ScrapedContent,
        format_type: str = "detailed",
        truncate: bool = True,
    ) -> str:
        """
        Format scraped content into different string formats.

        Args:
            content: ScrapedContent object
            format_type: Format type ('detailed', 'summary', 'minimal', 'markdown', 'full')
            truncate: Whether to truncate text in summary/minimal formats (default: True)

        Returns:
            Formatted string
        """
        if format_type == "detailed":
            return f"""URL: {content.url}
Title: {content.title}
Content: {content.text}
Links: {len(content.links)} found
Images: {len(content.images)} found
Word Count: {content.word_count}
Scraped at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(content.timestamp))}"""

        elif format_type == "full":
            return f"""URL: {content.url}
Title: {content.title}
Full Content: {content.text}
Links: {len(content.links)} found
Images: {len(content.images)} found
Word Count: {content.word_count}
Scraped at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(content.timestamp))}"""

        elif format_type == "summary":
            if truncate:
                return f"Title: {content.title}\nSummary: {content.text[:500]}{'...' if len(content.text) > 500 else ''}"
            else:
                return (
                    f"Title: {content.title}\nSummary: {content.text}"
                )

        elif format_type == "minimal":
            if truncate:
                return f"{content.title}: {content.text[:200]}{'...' if len(content.text) > 200 else ''}"
            else:
                return f"{content.title}: {content.text}"

        elif format_type == "markdown":
            return f"""# {content.title}

{content.text}

**Links:** {len(content.links)} found  
**Images:** {len(content.images)} found  
**Word Count:** {content.word_count}"""

        else:
            return content.text


# Convenience functions for easy use


def scrape_single_url_sync(url: str, **kwargs) -> ScrapedContent:
    """
    Synchronously scrape a single URL and return its structured content.

    This function creates a SuperFastScraper instance with any provided keyword arguments,
    scrapes the given URL, and returns a ScrapedContent object containing the extracted data.

    Args:
        url (str): The URL to scrape.
        **kwargs: Optional keyword arguments to configure the SuperFastScraper instance.
            Common options include:
                - timeout (int): Request timeout in seconds (default: 10)
                - max_retries (int): Number of retry attempts (default: 3)
                - user_agent (str): Custom User-Agent string
                - max_workers (int): Number of threads for concurrent requests
                - strip_html (bool): Whether to strip HTML tags from text (default: True)
                - remove_scripts (bool): Remove <script> tags (default: True)
                - remove_styles (bool): Remove <style> tags (default: True)
                - remove_comments (bool): Remove HTML comments (default: True)

    Returns:
        ScrapedContent: An object containing the URL, title, text, links, images, and metadata.

    Example:
        >>> content = scrape_single_url_sync("https://example.com")
        >>> print(content.title)
    """
    scraper = SuperFastScraper(**kwargs)
    return scraper.scrape_single_url(url)


def scrape_multiple_urls_sync(
    urls: List[str],
    format_type: str = "detailed",
    truncate: bool = True,
    **kwargs,
) -> str:
    """
    Synchronously scrape multiple URLs and return a formatted string of all results.

    This function creates a SuperFastScraper instance with any provided keyword arguments,
    scrapes each URL in the provided list concurrently, and returns a single formatted string
    containing the results for all URLs.

    Args:
        urls (List[str]): List of URLs to scrape.
        format_type (str): Output format for each result. Options:
            - 'detailed': Includes URL, title, content, links, images, word count, timestamp.
            - 'summary': Title and summary of content (optionally truncated).
            - 'minimal': Title and short snippet of content.
            - 'markdown': Markdown-formatted output.
            - 'full': Full content with all metadata.
        truncate (bool): Whether to truncate text in 'summary' and 'minimal' formats (default: True).
        **kwargs: Additional options for SuperFastScraper (see scrape_single_url_sync for details).

    Returns:
        str: Formatted string containing all scraped results, concatenated.

    Example:
        >>> urls = ["https://example.com", "https://httpbin.org/html"]
        >>> print(scrape_multiple_urls_sync(urls, format_type="summary"))
    """
    scraper = SuperFastScraper(**kwargs)
    return scraper.scrape_urls_formatted(urls, format_type, truncate)


def format_scraped_content(
    content: ScrapedContent,
    format_type: str = "detailed",
    truncate: bool = True,
) -> str:
    """
    Format a ScrapedContent object into a human-readable string.

    This function takes a ScrapedContent object and returns a string representation
    according to the specified format type.

    Args:
        content (ScrapedContent): The scraped content to format.
        format_type (str): Output format. Options:
            - 'detailed': Includes URL, title, content, links, images, word count, timestamp.
            - 'summary': Title and summary of content (optionally truncated).
            - 'minimal': Title and short snippet of content.
            - 'markdown': Markdown-formatted output.
            - 'full': Full content with all metadata.
        truncate (bool): Whether to truncate text in 'summary' and 'minimal' formats (default: True).

    Returns:
        str: Formatted string representation of the content.

    Example:
        >>> formatted = format_scraped_content(content, format_type="markdown")
        >>> print(formatted)
    """
    scraper = SuperFastScraper()
    return scraper.format_content(content, format_type, truncate)


def scrape_and_format_sync(
    url: str,
    format_type: str = "detailed",
    truncate: bool = True,
) -> str:
    """
    Synchronously scrape a URL and return its formatted content in a single call.

    This function combines scraping and formatting: it scrapes the given URL using
    SuperFastScraper (with any provided options), then formats the result according
    to the specified format type.

    Args:
        url (str): The URL to scrape.
        format_type (str): Output format. Options:
            - 'detailed': Includes URL, title, content, links, images, word count, timestamp.
            - 'summary': Title and summary of content (optionally truncated).
            - 'minimal': Title and short snippet of content.
            - 'markdown': Markdown-formatted output.
            - 'full': Full content with all metadata.
        truncate (bool): Whether to truncate text in 'summary' and 'minimal' formats (default: True).
        **kwargs: Additional options for SuperFastScraper (see scrape_single_url_sync for details).

    Returns:
        str: Formatted string content for the scraped URL.

    Example:
        >>> print(scrape_and_format_sync("https://example.com", format_type="summary"))
    """
    content = scrape_single_url_sync(url)
    return format_scraped_content(content, format_type, truncate)


# # Example usage
# if __name__ == "__main__":
#     urls = [
#         "https://swarms.ai",
#         "https://httpbin.org/html",
#         "https://httpbin.org/json",
#     ]

#     print(
#         scrape_multiple_urls_sync(
#             urls, "summary", truncate=True, max_workers=5
#         )
#     )
