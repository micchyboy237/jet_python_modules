import re
import time
import fnmatch
from jet.logger.timer import sleep_countdown
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


def sort_urls_numerically(urls):
    return sorted(urls, key=extract_numbers)


class SeleniumScraper:
    def __init__(self):
        self.driver: webdriver.Chrome = self._setup_driver()

    def _setup_driver(self) -> webdriver.Chrome:
        """Sets up the Chrome WebDriver with options."""
        chrome_options: Options = Options()
        driver: webdriver.Chrome = webdriver.Chrome(options=chrome_options)
        return driver

    def navigate_to_url(self, url: str):
        """Navigate the browser to the given URL, wait for the body element, and add extra delay."""
        self.driver.get(url)

        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        logger.info(f"Loaded {url}")

    def get_html(self, wait_time: int = 10) -> str:
        """Waits for full page load and returns the updated page source."""
        WebDriverWait(self.driver, wait_time).until(
            lambda driver: driver.execute_script(
                "return document.readyState") == "complete"
        )
        return self.driver.page_source

    def close(self):
        """Closes the browser and cleans up the driver."""
        self.driver.quit()


class WebCrawler(SeleniumScraper):
    def __init__(self, url: str = None, includes=None, excludes=None, includes_all=None, excludes_all=None, visited=None, max_depth: int = None):
        super().__init__()

        self.non_crawlable = ".php" in url
        self.base_url: str = self.normalize_url(url)
        self.host_name = urlparse(self.base_url).hostname
        self.visited_urls: set[str] = set(visited) if visited else set()
        self.passed_urls: set[str] = set()
        self.seen_urls: set[str] = {self.base_url}
        self.includes = includes or []
        self.excludes = excludes or []
        self.includes_all = includes_all or []  # NEW: AND condition includes
        self.excludes_all = excludes_all or []  # NEW: AND condition excludes
        # NEW: Optional max depth (None = unlimited)
        self.max_depth = max_depth

    def change_url(self, url: str):
        try:
            self.navigate_to_url(url)
        except Exception as e:
            logger.error(f"Error navigating to URL {url}: {e}")

    def crawl(self, url: str):
        yield from self._crawl_recursive(url, depth=0)

    def _crawl_recursive(self, url: str, depth: int):
        if self.max_depth is not None and depth > self.max_depth:
            logger.info(f"Max depth reached for {url}")
            return

        url = self.normalize_url(url)
        if url in self.visited_urls:
            return

        self.visited_urls.add(url)

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

        full_urls = set()
        for anchor in anchor_elements:
            try:
                full_url = anchor.get_attribute("href")
                if full_url:
                    full_url = self.normalize_url(full_url)
                    if self.host_name in full_url and not self._should_exclude(full_url):
                        full_urls.add(full_url)
            except Exception as e:
                logger.error(f"Failed to process anchor: {e}")
                continue

        for full_url in sort_urls_numerically(list(full_urls)):
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

    for start_url in urls:
        crawler = WebCrawler(start_url,
                             excludes=excludes, includes_all=includes_all, max_depth=max_depth)

        output_file = f"generated/crawl/{crawler.host_name}_urls.json"
        batch_size = 5
        batch_count = 0

        results = []
        for result in crawler.crawl(crawler.base_url):
            logger.info(
                f"Saving {len(crawler.passed_urls)} pages to {output_file}")
            results.append(result)
            save_data(output_file, results, write=True)

        crawler.close()
