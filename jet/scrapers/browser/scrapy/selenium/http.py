"""This module contains the ``SeleniumRequest`` class"""

from scrapy import Request


class SeleniumRequest(Request):
    """Scrapy ``Request`` subclass providing additional arguments"""

    def __init__(self, wait_time=None, wait_until=None, screenshot=False, script=None, scroll_bottom=False, always_restart=False, *args, **kwargs):
        """Initialize a new selenium request

        Parameters
        ----------
        wait_time: int
            The number of seconds to wait.
        wait_until: method
            One of the "selenium.webdriver.support.expected_conditions". The response
            will be returned until the given condition is fulfilled.
        screenshot: bool or str
            If True, a screenshot of the page will be taken and the data of the screenshot
            will be returned in the response "meta" attribute.
            If a string is given, the screenshot will be saved to the given path.
        script: str
            JavaScript code to execute.
        scroll_bottom: bool
            If True, the page will be scrolled to the bottom before returning the response.
        always_restart: bool
            If True, the driver will be restarted for each request.
        """

        self.wait_time = wait_time
        self.wait_until = wait_until
        self.screenshot = screenshot
        self.script = script
        self.scroll_bottom = scroll_bottom
        self.always_restart = always_restart

        super().__init__(*args, **kwargs)


__all__ = [
    "SeleniumRequest",
]
