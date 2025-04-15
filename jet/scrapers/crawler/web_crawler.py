import re
import time
import fnmatch
from typing import Generator, Optional, TypedDict
from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.logger.timer import sleep_countdown
from jet.scrapers.preprocessor import html_to_markdown
from jet.scrapers.utils import scrape_links
from jet.utils.url_utils import normalize_url
from jet.wordnet.similarity import query_similarity_scores
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse, unquote, urlunparse
from fake_useragent import UserAgent
from jet.wordnet.sentence import split_sentences
from jet.file.utils import save_data
from jet.logger import logger
from tqdm import tqdm


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


def get_headers_from_html(html: str) -> list[str]:
    md_text = html_to_markdown(html, ignore_links=False)
    header_contents = get_md_header_contents(md_text)
    headers = [header["content"] for header in header_contents]
    return headers


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
        # Prevents sites from detecting Selenium automation.
        chrome_options.add_argument(
            "--disable-blink-features=AutomationControlled")
        # Ensure a clean profile for performance (optional)
        chrome_options.add_argument("--incognito")
        # Set user-agent to avoid bot detection
        ua = UserAgent()
        random_user_agent = ua.random
        chrome_options.add_argument(f"--user-agent={random_user_agent}")

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
        self.base_url: str = normalize_url(url) if url else ""
        self.host_name = urlparse(self.base_url).hostname
        self.base_host_url = f"{urlparse(self.base_url).scheme}://{self.host_name}"
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
        self.top_n = 5

    def change_url(self, url: str):
        try:
            self.navigate_to_url(url)
        except Exception as e:
            logger.error(f"Error navigating to URL {url}: {e}")

    def crawl(self, url: str):
        self.non_crawlable = ".php" in url
        self.base_url: str = self._normalize_url(url)
        self.host_name = urlparse(self.base_url).hostname
        self.seen_urls: set[str] = {self.base_url}

        yield from self._crawl_recursive(url, depth=0)

    def _crawl_recursive(self, url: str, depth: int) -> Generator[PageResult, None, None]:
        if self.max_depth is not None and depth > self.max_depth:
            # logger.info(f"Max depth reached for {url}")
            return

        url = self._normalize_url(url)
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

        all_links = scrape_links(html_str)
        full_urls = set()
        seen_keys = set()

        for link in all_links:
            try:
                if not link or link.startswith("#"):
                    continue

                link = self._normalize_url(link)

                if self.base_url == link:
                    continue

                parsed = urlparse(link)

                if link.startswith(self.base_host_url) or parsed.hostname is None:
                    if self._should_exclude(link):
                        continue

                    # Create unique key using hostname + path only
                    # Normalize path
                    key = (parsed.hostname, parsed.path.rstrip("/"))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        # Rebuild URL with only scheme, hostname, path
                        clean_url = urlunparse(
                            (parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
                        full_urls.add(clean_url)

            except Exception as e:
                logger.error(f"Failed to process anchor: {e}")
                continue

        full_urls = list(full_urls)

        if full_urls:
            full_urls = sort_urls_numerically(full_urls)

            if self.query:
                search_link_results = query_similarity_scores(
                    self.query, full_urls)
                relevant_urls = [result["text"]
                                 for result in search_link_results]
            else:
                relevant_urls = full_urls

            for relevant_url in relevant_urls:
                yield from self._crawl_recursive(relevant_url, depth=depth + 1)

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

    def _normalize_url(self, url: str) -> str:
        return normalize_url(url, self.base_url)


# Example usage
if __name__ == "__main__":
    from jet.scrapers.utils import search_data

    query = "Philippines tips for online selling 2025"
    max_search_depth = 1

    search_results = search_data(query)
    urls = [item["url"] for item in search_results]

    max_depth = None

    crawler = WebCrawler(max_depth=max_depth, query=query)

    selected_html = []
    pbar = tqdm(total=len(urls))
    for url in urls:
        domain = urlparse(url).netloc
        pbar.set_description(f"Domain: {domain}")
        pbar.update(1)

        for result in crawler.crawl(url):
            selected_html.append((result["url"], result["html"]))

    crawler.close()
