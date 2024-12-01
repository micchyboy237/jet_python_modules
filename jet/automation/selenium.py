from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from fake_useragent import UserAgent
from typing import Optional


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
            f"user-agent={self._get_random_user_agent()}")
        return webdriver.Chrome(options=chrome_options)

    def _get_random_user_agent(self) -> str:
        """Generates a random user agent string."""
        return UserAgent().random

    def navigate_to_url(self, url: str):
        """Navigate the browser to the given URL."""
        self.driver.get(url)
        # Initial wait to load html
        self.wait_for_element("title", 10)

    def get_page_source(self) -> str:
        """Returns the page source (HTML) as a string."""
        return self.driver.page_source

    def wait_for_element(self, css_selector: str, duration: int) -> Optional[webdriver.remote.webelement.WebElement]:
        """Waits for an element to be present on the page and returns it."""
        try:
            return WebDriverWait(self.driver, duration).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, css_selector))
            )
        except Exception as e:
            print(f"Error: {e}")
            return None

    def close(self):
        """Closes the browser and cleans up the driver."""
        self.driver.quit()
