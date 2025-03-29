"""This module contains the ``SeleniumMiddleware`` scrapy middleware"""

import jet.scrapers.browser.scrapy.settings.config
import logging

from importlib import import_module
from queue import Queue
from shutil import which
from threading import Lock
from scrapy import signals
from scrapy.exceptions import NotConfigured
from scrapy.http import HtmlResponse
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options

from .http import SeleniumRequest


class SeleniumMiddleware:
    """Scrapy middleware handling the requests using selenium"""

    def __init__(self, driver_name, driver_executable_path,
                 browser_executable_path, command_executor, driver_arguments, driver_count):
        """Initialize the selenium webdriver pool

        Parameters
        ----------
        driver_name: str
            The selenium ``WebDriver`` to use
        driver_executable_path: str
            The path of the executable binary of the driver
        driver_arguments: list
            A list of arguments to initialize the driver
        browser_executable_path: str
            The path of the executable binary of the browser
        command_executor: str
            Selenium remote server endpoint
        driver_count: int
            Number of WebDriver instances to initialize
        """

        self.driver_queue = Queue(maxsize=driver_count)
        self.lock = Lock()

        for _ in range(driver_count):
            driver = self._create_driver(driver_name, driver_executable_path,
                                         browser_executable_path, command_executor, driver_arguments)
            self.driver_queue.put(driver)

    def _create_driver(self, driver_name, driver_executable_path,
                       browser_executable_path, command_executor, driver_arguments):
        """Create a new selenium webdriver instance"""

        webdriver_base_path = f'selenium.webdriver.{driver_name}'

        driver_klass_module = import_module(f'{webdriver_base_path}.webdriver')
        driver_klass = getattr(driver_klass_module, 'WebDriver')

        driver_options_module = import_module(f'{webdriver_base_path}.options')
        driver_options_klass = getattr(driver_options_module, 'Options')

        driver_options: Options = driver_options_klass()

        if browser_executable_path:
            driver_options.binary_location = browser_executable_path
        for argument in driver_arguments:
            driver_options.add_argument(argument)

        # remote driver
        if command_executor is not None:
            driver = webdriver.Remote(command_executor=command_executor,
                                      options=driver_options)
        # locally installed driver
        elif driver_executable_path is not None:
            service_module = import_module(f'{webdriver_base_path}.service')
            service_klass = getattr(service_module, 'Service')
            service_kwargs = {
                'executable_path': driver_executable_path,
            }
            service = service_klass(**service_kwargs)
            driver_kwargs = {
                'service': service,
                'options': driver_options
            }
            driver = driver_klass(**driver_kwargs)

        return driver

    @classmethod
    def from_crawler(cls, crawler):
        """Initialize the middleware with the crawler settings"""

        driver_name = crawler.settings.get('SELENIUM_DRIVER_NAME', 'chrome')
        driver_executable_path = crawler.settings.get(
            'SELENIUM_DRIVER_EXECUTABLE_PATH', which('chromedriver'))
        browser_executable_path = crawler.settings.get(
            'SELENIUM_BROWSER_EXECUTABLE_PATH')
        command_executor = crawler.settings.get('SELENIUM_COMMAND_EXECUTOR')
        driver_arguments = crawler.settings.get('SELENIUM_DRIVER_ARGUMENTS', [
            '--headless=new',  # Comment this out to allow the browser to open
            '--no-sandbox',
            '--disable-gpu',
            '--window-size=1512,982',  # MacBook M1 default viewport size
            '--force-device-scale-factor=0.8',  # Set zoom to 80%
            '--disable-blink-features',
            '--disable-blink-features=AutomationControlled',
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        ])
        driver_count = crawler.settings.get('SELENIUM_DRIVER_COUNT', 1)

        if driver_name is None:
            raise NotConfigured('SELENIUM_DRIVER_NAME must be set')

        if driver_executable_path is None and command_executor is None:
            raise NotConfigured('Either SELENIUM_DRIVER_EXECUTABLE_PATH '
                                'or SELENIUM_COMMAND_EXECUTOR must be set')

        middleware = cls(
            driver_name=driver_name,
            driver_executable_path=driver_executable_path,
            browser_executable_path=browser_executable_path,
            command_executor=command_executor,
            driver_arguments=driver_arguments,
            driver_count=driver_count
        )

        crawler.signals.connect(
            middleware.spider_closed, signals.spider_closed)

        return middleware

    @staticmethod
    def scroll_down_until_no_more_content(driver, timeout=10):
        last_height = driver.execute_script(
            "return document.body.scrollHeight")

        while True:
            driver.execute_script(
                "window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });")

            try:
                WebDriverWait(driver, timeout).until(
                    lambda d: d.execute_script(
                        "return document.body.scrollHeight") > last_height
                )
            except TimeoutException:
                break

            new_height = driver.execute_script(
                "return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def _restart_driver(self, driver):
        old_driver = driver
        driver_options = driver.options

        # locally installed driver
        if driver.service:
            service = driver.service
            driver_kwargs = {
                'service': service,
                'options': driver_options
            }
            new_driver = driver.__class__(**driver_kwargs)
        # remote driver
        elif driver.command_executor:
            new_driver = webdriver.Remote(command_executor=driver.command_executor._url,
                                          options=driver_options)
        old_driver.quit()
        return new_driver

    def process_request(self, request, spider):
        """Process a request using the selenium driver if applicable"""

        if not isinstance(request, SeleniumRequest):
            return None

        with self.lock:
            driver = self.driver_queue.get()

        if request.always_restart:
            driver = self._restart_driver(driver)

        driver.get(request.url)

        for cookie_name, cookie_value in request.cookies.items():
            driver.add_cookie(
                {
                    'name': cookie_name,
                    'value': cookie_value
                }
            )

        if request.wait_until:
            try:
                WebDriverWait(driver, request.wait_time).until(
                    request.wait_until
                )
            except TimeoutException:
                pass

        if isinstance(request.screenshot, bool) and request.screenshot:
            request.meta['screenshot'] = driver.get_screenshot_as_png()
        elif isinstance(request.screenshot, str):
            screenshot_data = driver.get_screenshot_as_png()
            with open(request.screenshot, 'wb') as f:
                f.write(screenshot_data)
            request.meta['screenshot'] = request.screenshot

        if request.script:
            try:
                driver.execute_script(request.script)
            except Exception as e:
                logging.error(f"JavaScript execution error: {e}")

        if request.scroll_bottom:
            SeleniumMiddleware.scroll_down_until_no_more_content(driver)

        body = str.encode(driver.page_source)

        # Expose the driver via the "meta" attribute
        request.meta.update({'driver': driver})

        response = HtmlResponse(
            driver.current_url,
            body=body,
            encoding='utf-8',
            request=request
        )

        with self.lock:
            self.driver_queue.put(driver)

        return response

    def spider_closed(self):
        """Shutdown all drivers when spider is closed"""

        while not self.driver_queue.empty():
            driver = self.driver_queue.get()
            driver.quit()


__all__ = [
    "SeleniumMiddleware",
]
