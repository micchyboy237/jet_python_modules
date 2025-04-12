import re
import time
import fnmatch
from typing import Generator, Optional, TypedDict
from jet.logger.timer import sleep_countdown
from jet.scrapers.utils import scrape_links
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse, unquote
from jet.file.utils import save_data
from jet.logger import logger


def extract_numbers(text):
    """Extract numbers from the URL and use them as sorting keys."""
    parts = re.split(r'(\d+)', text)
    return [int(part) if part.isdigit() else part for part in parts]


def count_path_segments(url):
    """Count the number of path segments in the URL."""
    path = urlparse(url).path
    return len([segment for segment in path.split('/') if segment])


def sort_urls_numerically(urls):
    return sorted(urls, key=lambda url: (count_path_segments(url), extract_numbers(url)))


class SeleniumScraper:
    def __init__(self, max_retries: int = 5):
        self.driver: webdriver.Chrome = self._setup_driver()
        self.max_retries = max_retries

    def _setup_driver(self) -> webdriver.Chrome:
        """Sets up the Chrome WebDriver with optimized options for reliability and speed."""
        chrome_options: Options = Options()

        # Use "eager" to start interacting with the page sooner (use "normal" if required)
        chrome_options.page_load_strategy = "eager"

        # Reduce resource load (optional: remove headless mode if debugging)
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--no-sandbox")  # For Linux servers
        # Prevent crashes on low-memory systems
        chrome_options.add_argument("--disable-dev-shm-usage")

        # Ensure a clean profile for performance (optional)
        chrome_options.add_argument("--incognito")

        # Set user-agent to avoid bot detection
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )

        # Initialize the driver with error handling
        try:
            driver: webdriver.Chrome = webdriver.Chrome(options=chrome_options)
            # Increase timeout to handle slow pages
            driver.set_page_load_timeout(60)
            # Increase JavaScript execution timeout
            driver.set_script_timeout(30)
            return driver
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise

    def navigate_to_url(self, url: str):
        """Navigate the browser to the given URL, retrying with exponential backoff."""
        retries = 0
        while retries < self.max_retries:
            try:
                self.driver.get(url)
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                logger.info(f"Loaded {url}")
                return
            except Exception as e:
                retries += 1
                # Exponential backoff (2^retries seconds)
                wait_time = 2 ** retries
                logger.error(
                    f"Error loading {url}, attempt {retries}/{self.max_retries}: {e}")
                if retries < self.max_retries:
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Max retries reached for {url}")
                    raise

    def get_html(self, wait_time: int = 10) -> str:
        """Waits for full page load and returns the updated page source."""
        retries = 0
        while retries < self.max_retries:
            try:
                return self.driver.page_source
            except Exception as e:
                retries += 1
                # Exponential backoff (2^retries seconds)
                wait_time = 2 ** retries
                logger.error(
                    f"Error getting HTML, attempt {retries}/{self.max_retries}: {e}")
                if retries < self.max_retries:
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached for getting HTML")
                    raise

    def close(self):
        """Closes the browser and cleans up the driver."""
        self.driver.quit()


class PageResult(TypedDict):
    url: str
    html: str


class WebCrawler(SeleniumScraper):
    def __init__(self, url: Optional[str] = None, includes=None, excludes=None, includes_all=None, excludes_all=None, visited=None, max_depth: Optional[int] = None, query: Optional[str] = None, max_visited: Optional[int] = None, max_retries: int = 5):
        super().__init__(max_retries=max_retries)

        self.non_crawlable = False
        self.base_url: str = self.normalize_url(url)
        self.host_name = urlparse(self.base_url).hostname
        self.seen_urls: set[str] = {self.base_url}
        self.visited_urls: set[str] = set(visited) if visited else set()
        self.passed_urls: set[str] = set()
        self.includes = includes or []
        self.excludes = excludes or []
        self.includes_all = includes_all or []  # NEW: AND condition includes
        self.excludes_all = excludes_all or []  # NEW: AND condition excludes
        # max depth (None = unlimited)
        self.max_depth = max_depth
        self.max_visited = max_visited
        self.query = query

    def change_url(self, url: str):
        try:
            self.navigate_to_url(url)
        except Exception as e:
            logger.error(f"Error navigating to URL {url}: {e}")

    def crawl(self, url: str):
        self.non_crawlable = ".php" in url
        self.base_url: str = self.normalize_url(url)
        self.host_name = urlparse(self.base_url).hostname
        self.seen_urls: set[str] = {self.base_url}

        yield from self._crawl_recursive(url, depth=0)

    def _crawl_recursive(self, url: str, depth: int) -> Generator[PageResult, None, None]:
        if self.max_depth is not None and depth > self.max_depth:
            # logger.info(f"Max depth reached for {url}")
            return

        url = self.normalize_url(url)
        if url in self.visited_urls:
            return

        self.visited_urls.add(url)

        if self.max_visited and len(self.visited_urls) >= self.max_visited:
            return

        logger.info(f"Crawling (Depth {depth}): {url}")
        self.change_url(url)

        if url != self.base_url and self._should_exclude(url):
            return

        self.passed_urls.add(url)
        html_str = self.get_html()
        yield {
            "url": url,
            "html": html_str,
        }

        extension = url.split('.')[-1].lower()
        if extension in ["pdf", "doc", "docx", "ppt", "pptx"]:
            return

        try:
            sleep_countdown(1)
            anchor_elements = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.TAG_NAME, "a"))
            )
        except Exception as e:
            logger.error(f"Failed to find anchor elements: {e}")
            return

        if self.non_crawlable:
            return

        links = scrape_links(html_str)
        full_urls = set()
        for link in links:
            try:
                if link:
                    link = self.normalize_url(link)
                    parsed_link = urlparse(link)
                    if (
                        link.startswith(self.base_url)
                        or parsed_link.hostname is None
                    ):
                        if not self._should_exclude(link):
                            full_urls.add(link)
            except Exception as e:
                logger.error(f"Failed to process anchor: {e}")
                continue

        sorted_full_urls = sort_urls_numerically(list(full_urls))
        for full_url in sorted_full_urls:
            yield from self._crawl_recursive(full_url, depth=depth + 1)

    def _should_exclude(self, url: str) -> bool:
        """Check if a URL should be excluded based on filtering rules."""
        failed_includes = self.includes and not any(
            fnmatch.fnmatch(url.lower(), pattern.lower()) for pattern in self.includes)
        failed_excludes = self.excludes and any(
            fnmatch.fnmatch(url.lower(), pattern.lower()) for pattern in self.excludes)

        failed_includes_all = self.includes_all and not all(
            fnmatch.fnmatch(url.lower(), pattern.lower()) for pattern in self.includes_all)
        failed_excludes_all = self.excludes_all and all(
            fnmatch.fnmatch(url.lower(), pattern.lower()) for pattern in self.excludes_all)

        return failed_includes or failed_excludes or failed_includes_all or failed_excludes_all

    def normalize_url(self, url: str) -> str:
        if not url:
            return ""

        if self.non_crawlable:
            return url

        parsed = urlparse(url)

        # If the URL is relative, append it to base URL
        if not parsed.scheme or not parsed.netloc:
            return f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"

        normalized_url = parsed.scheme + "://" + parsed.netloc + parsed.path
        return unquote(normalized_url.rstrip('/'))


# Example usage
if __name__ == "__main__":
    urls = [
        "https://reelgood.com/show/ill-become-a-villainess-who-goes-down-in-history-2024",
    ]

    includes_all = ["*villainess*", "*down*", "*history*"]
    excludes = []
    max_depth = None

    crawler = WebCrawler(
        excludes=excludes, includes_all=includes_all, max_depth=max_depth)

    for start_url in urls:
        batch_size = 5
        batch_count = 0

        results = []
        for result in crawler.crawl(start_url):
            output_file = f"generated/crawl/{crawler.host_name}_urls.json"
            logger.info(
                f"Saving {len(crawler.passed_urls)} pages to {output_file}")
            results.append(result)
            save_data(output_file, results, write=True)

    crawler.close()
