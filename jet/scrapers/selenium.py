import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from fake_useragent import UserAgent
from parsel import Selector
from typing import Optional, List


class SeleniumScraper:
    def __init__(self, headless: bool = False):
        """Initialize the SeleniumScraper with an optional headless mode."""
        self.driver = self._setup_driver(headless)

    def _setup_driver(self, headless: bool) -> webdriver.Chrome:
        """Sets up the Chrome WebDriver with options."""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument(
            f"user-agent={self._get_random_user_agent()}"
        )
        return webdriver.Chrome(options=chrome_options)

    def _get_random_user_agent(self) -> str:
        """Generates a random user agent string."""
        return UserAgent().random

    def navigate_to_url(self, url: str):
        """Navigate the browser to the given URL."""
        self.driver.get(url)
        self.wait_for_element("title", 10)
        self.wait_for_page_dom_load(10)
        self.wait_for_async_calls_complete(10)
        # Slight delay
        # time.sleep(2)

    def get_page_source(self) -> str:
        """Returns the page source (HTML) as a string."""
        return self.driver.page_source

    def wait_for_element(self, css_selector: str, duration: int):
        """Waits for an element to be present on the page."""
        WebDriverWait(self.driver, duration).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, css_selector))
        )

    def wait_for_page_dom_load(self, timeout: int):
        """Waits until the page is fully loaded."""
        try:
            WebDriverWait(self.driver, timeout).until(
                lambda d: d.execute_script(
                    "return document.readyState") == "complete"
            )
        except Exception as e:
            print(f"Error waiting for page to load: {e}")

    def wait_for_async_calls_complete(self, timeout: int):
        """
        Waits for all network requests to complete, even on non-jQuery websites.

        Args:
            timeout (int): Maximum time to wait.
        """
        try:
            WebDriverWait(self.driver, timeout).until(
                lambda d: d.execute_script("""
                    return (
                        window.performance &&
                        window.performance.getEntriesByType('resource').every(
                            e => !!e.responseEnd
                        )
                    );
                """)
            )
        except Exception as e:
            print(f"Error waiting for AJAX to complete: {e}")

    def wait_for_text_content(self, css_selector: str, timeout: int) -> Optional[str]:
        """
        Wait for specific text content to load within the selected element.

        Args:
            css_selector (str): The CSS selector of the element to wait for.
            timeout (int): Maximum time to wait.

        Returns:
            Optional[str]: The text content of the element or None if timeout.
        """
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, css_selector))
            )
            return element.text if element else None
        except Exception as e:
            print(f"Error waiting for text content: {e}")
            return None

    def get_all_text(self) -> str:
        """Retrieve all text content from the page."""
        try:
            return self.driver.find_element(By.TAG_NAME, "body").text
        except Exception as e:
            print(f"Error retrieving text: {e}")
            return ""

    def close(self):
        """Closes the browser and cleans up the driver."""
        self.driver.quit()


class UrlScraper:
    def __init__(self) -> None:
        self.scraper = SeleniumScraper()

    def scrape_url(self, url: str) -> str:
        self.scraper.navigate_to_url(url)
        html_str = self.scraper.get_page_source()
        self.scraper.close()
        return html_str
