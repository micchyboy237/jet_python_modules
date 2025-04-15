from threading import Lock
import re
import time
import fnmatch
from typing import AsyncGenerator, Generator, Optional, TypedDict, List
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
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# [Existing functions: extract_numbers, count_path_segments, sort_urls_numerically, get_headers_from_html]


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
        chrome_options.page_load_strategy = "eager"
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(
            "--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--incognito")
        ua = UserAgent()
        random_user_agent = ua.random
        chrome_options.add_argument(f"--user-agent={random_user_agent}")
        try:
            driver: webdriver.Chrome = webdriver.Chrome(options=chrome_options)
            # driver.set_page_load_timeout(60)
            # driver.set_script_timeout(30)
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
    def __init__(self, urls: Optional[List[str]] = None, includes=None, excludes=None,
                 includes_all=None, excludes_all=None, visited=None, max_depth: Optional[int] = 0,
                 query: Optional[str] = None, max_visited: Optional[int] = None,
                 max_retries: int = 5, max_parallel: int = 5):
        super().__init__(max_retries=max_retries)
        if max_visited is not None and max_visited <= 0:
            raise ValueError("max_visited must be a positive integer or None")
        self.non_crawlable = False
        self.base_urls: List[str] = [normalize_url(
            url) for url in urls] if urls else []
        self.host_names = {urlparse(url).hostname for url in self.base_urls}
        self.base_host_urls = {
            f"{urlparse(url).scheme}://{urlparse(url).hostname}" for url in self.base_urls}
        self.seen_urls: set[str] = set(self.base_urls)
        self.visited_urls: set[str] = set(visited) if visited else set()
        self.visited_urls_lock = Lock()
        self.passed_urls: set[str] = set()
        self.includes = includes or []
        self.excludes = excludes or []
        self.includes_all = includes_all or []
        self.excludes_all = excludes_all or []
        self.max_depth = max_depth
        self.max_visited = max_visited
        self.query = query
        self.top_n = 5
        self.max_parallel = max_parallel
        self.executor = ThreadPoolExecutor(max_workers=max_parallel)

    def change_url(self, url: str):
        try:
            self.navigate_to_url(url)
        except Exception as e:
            logger.error(f"Error navigating to URL {url}: {e}")

    async def crawl(self) -> AsyncGenerator[PageResult, None]:
        """Crawl all URLs (initial and child) in parallel, yielding results asynchronously."""
        if not self.base_urls:
            return
        async for result in self._crawl_async():
            yield result

    async def _crawl_async(self) -> AsyncGenerator[PageResult, None]:
        """Asynchronous wrapper to manage parallel crawling of all URLs."""
        pending_urls = [(url, 0) for url in self.base_urls]  # (url, depth)
        tasks = []

        # Initialize progress bar
        pbar = tqdm(total=len(self.base_urls) if self.max_visited is None else min(self.max_visited, len(self.base_urls)),
                    desc="Crawling URLs", unit="page")

        # Start initial tasks
        while len(tasks) < self.max_parallel and pending_urls:
            url, depth = pending_urls.pop(0)
            task = asyncio.get_event_loop().run_in_executor(
                self.executor,
                partial(self._process_url, url, depth)
            )
            tasks.append((task, url, depth))

        # Process tasks dynamically
        while tasks:
            completed, pending = await asyncio.wait(
                [task for task, _, _ in tasks], return_when=asyncio.FIRST_COMPLETED
            )
            new_urls = []
            for task in completed:
                try:
                    result, child_urls = await task
                    if result:
                        with self.visited_urls_lock:
                            visited_count = len(self.visited_urls)
                            max_visited_str = f"/{self.max_visited}" if self.max_visited else ""
                            pbar.set_description(
                                f"Visited {visited_count}{max_visited_str} | Domain: {urlparse(result['url']).netloc}")
                            pbar.update(1)
                            yield result
                        new_urls.extend((child_url, depth + 1)
                                        for child_url in child_urls)
                except Exception as e:
                    logger.error(f"Task failed: {e}")
            tasks = [(task, url, depth)
                     for task, url, depth in tasks if task not in completed]
            # Add new URLs (prioritize child URLs to maintain depth-first behavior)
            pending_urls = new_urls + pending_urls
            while len(tasks) < self.max_parallel and pending_urls:
                url, depth = pending_urls.pop(0)
                with self.visited_urls_lock:
                    if url not in self.visited_urls and (self.max_visited is None or len(self.visited_urls) < self.max_visited):
                        task = asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            partial(self._process_url, url, depth)
                        )
                        tasks.append((task, url, depth))

        pbar.close()

    def _process_url_task(self, url: str, depth: int) -> tuple[Optional[PageResult], List[str]]:
        """Process a single URL in a separate thread."""
        try:
            return self._process_url(url, depth)
        finally:
            pass  # SeleniumScraper cleanup handled by the main instance

    def _process_url(self, url: str, depth: int) -> tuple[Optional[PageResult], List[str]]:
        """Process a single URL and return its result and child URLs."""
        if self.max_depth is not None and depth > self.max_depth:
            return None, []
        url = self._normalize_url(url)
        with self.visited_urls_lock:
            if url in self.visited_urls:
                return None, []
            if self.max_visited and len(self.visited_urls) >= self.max_visited:
                return None, []
            self.visited_urls.add(url)

        logger.info(f"Crawling (Depth {depth}): {url}")
        try:
            self.change_url(url)
        except Exception as e:
            logger.error(f"Failed to navigate to {url}: {e}")
            return None, []

        if url not in self.base_urls and self._should_exclude(url):
            return None, []

        self.passed_urls.add(url)
        try:
            html_str = self.get_html()
        except Exception as e:
            logger.error(f"Failed to get HTML for {url}: {e}")
            return None, []

        result = {"url": url, "html": html_str}

        if depth + 1 > self.max_depth:
            return result, []

        try:
            sleep_countdown(1)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.TAG_NAME, "a"))
            )
        except Exception as e:
            logger.error(f"Failed to find anchor elements for {url}: {e}")
            return result, []

        if self.non_crawlable:
            return result, []

        all_links = scrape_links(html_str)
        full_urls = set()
        seen_keys = set()

        for link in all_links:
            try:
                if not link or link.startswith("#"):
                    continue
                link = self._normalize_url(link)
                if link in self.base_urls:
                    continue
                parsed = urlparse(link)
                if any(link.startswith(base_host_url) for base_host_url in self.base_host_urls) or parsed.hostname is None:
                    if self._should_exclude(link):
                        continue
                    key = (parsed.hostname, parsed.path.rstrip("/"))
                    if key not in seen_keys:
                        seen_keys.add(key)
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
        else:
            relevant_urls = []

        return result, relevant_urls

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
        """Normalize a URL relative to the first base URL."""
        base_url = self.base_urls[0] if self.base_urls else ""
        return normalize_url(url, base_url)

    def close(self):
        """Closes the browser, executor, and cleans up resources."""
        super().close()
        self.executor.shutdown(wait=True)


# Example usage
if __name__ == "__main__":
    from jet.scrapers.utils import search_data

    query = "Philippines tips for online selling 2025"
    max_search_depth = 0

    search_results = search_data(query)
    urls = [item["url"] for item in search_results][:6]

    crawler = WebCrawler(urls=urls, query=query,
                         max_visited=5, max_depth=max_search_depth)

    async def main():
        selected_html = []
        async for result in crawler.crawl():
            logger.success(f"Done crawling {result['url']}")
            selected_html.append((result['url'], result['html']))
        return selected_html

    selected_html = asyncio.run(main())
    crawler.close()
    print(f"Crawled {len(selected_html)} pages")
