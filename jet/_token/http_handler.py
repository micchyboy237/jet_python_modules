import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class HTTPHandler:
    def __init__(self, concurrent_limit=1, retries=3, backoff_factor=0.3):
        self.concurrent_limit = concurrent_limit
        self.session = requests.Session()

        # Set up retries for robust handling
        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get(self, url: str):
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        return response

    def close(self):
        self.session.close()
