# video_downloader.py
from typing import List, Optional, Literal
from pathlib import Path
import logging
import requests
import m3u8
import ffmpeg
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoDownloader:
    """A class to download and merge HLS streaming videos."""

    def __init__(self, output_dir: str = "downloads"):
        """Initialize the downloader with an output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session = requests.Session()

    def fetch_playlist(self, m3u8_url: str) -> Optional[m3u8.M3U8]:
        """Fetch and parse the m3u8 playlist."""
        try:
            response = self.session.get(
                m3u8_url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            playlist = m3u8.loads(response.text)
            if playlist.is_variant:
                # Select the highest quality variant
                highest_bandwidth = max(
                    (v for v in playlist.playlists if v.stream_info.bandwidth),
                    key=lambda v: v.stream_info.bandwidth,
                    default=None
                )
                if highest_bandwidth:
                    return self.fetch_playlist(highest_bandwidth.absolute_uri)
            return playlist
        except requests.RequestException as e:
            logger.error(f"Failed to fetch playlist: {e}")
            return None

    # Other methods (download_segments, merge_segments, cleanup, download_video) remain unchanged


class SeleniumDownloader:
    """A class to automate video stream detection and downloading using Selenium."""

    def __init__(self, downloader: VideoDownloader):
        """Initialize with a VideoDownloader instance."""
        self.downloader = downloader
        self.driver: Optional[webdriver.Chrome] = None

    def setup_driver(self) -> None:
        """Set up a headless Chrome driver."""
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

        # Enable network logging
        capabilities = webdriver.ChromeOptions().to_capabilities()
        capabilities["goog:loggingPrefs"] = {"performance": "ALL"}

        try:
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options
            )
            logger.info("Chrome driver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Chrome driver: {e}")
            raise

    def close_driver(self) -> None:
        """Close the Chrome driver."""
        if self.driver:
            self.driver.quit()
            logger.info("Chrome driver closed")

    def get_m3u8_url(self, video_url: str, timeout: int = 30) -> Optional[str]:
        """Detect the m3u8 URL from a running video page."""
        if not self.driver:
            self.setup_driver()

        try:
            self.driver.get(video_url)
            # Wait for video player to load
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "video"))
            )

            # Capture network logs
            logs = self.driver.get_log("performance")
            for entry in logs:
                log = json.loads(entry["message"])["message"]
                if log["method"] == "Network.responseReceived":
                    url = log["params"]["response"]["url"]
                    if url.endswith(".m3u8"):
                        logger.info(f"Found m3u8 URL: {url}")
                        return url

            logger.error("No m3u8 URL found in network logs")
            return None
        except Exception as e:
            logger.error(f"Failed to detect m3u8 URL: {e}")
            return None
        finally:
            self.close_driver()

    def download_running_video(self, video_url: str, output_file: str) -> bool:
        """Automatically download a running video from a given URL."""
        m3u8_url = self.get_m3u8_url(video_url)
        if not m3u8_url:
            logger.error("Could not detect video stream URL")
            return False

        return self.downloader.download_video(m3u8_url, output_file)
