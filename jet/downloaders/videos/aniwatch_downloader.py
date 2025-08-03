from typing import List, Optional, Literal, Dict
from pathlib import Path
import requests
import m3u8
import ffmpeg
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from jet.logger import logger


class VideoDownloader:
    """A class to download and merge streaming videos."""

    def __init__(self, output_dir: str = "downloads"):
        """Initialize the downloader with an output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"})

    def fetch_playlist(self, m3u8_url: str) -> Optional[m3u8.M3U8]:
        """Fetch and parse the m3u8 playlist, handling variant playlists."""
        try:
            response = self.session.get(m3u8_url)
            response.raise_for_status()
            playlist = m3u8.loads(response.text)
            if playlist.is_variant:
                highest_bandwidth = max(
                    (v for v in playlist.playlists if v.stream_info.bandwidth),
                    key=lambda v: v.stream_info.bandwidth,
                    default=None
                )
                if highest_bandwidth:
                    # Ensure base_uri is set for relative URLs
                    if not highest_bandwidth.base_uri:
                        highest_bandwidth.base_uri = m3u8_url[:m3u8_url.rfind(
                            "/") + 1]
                    return self.fetch_playlist(highest_bandwidth.absolute_uri)
            return playlist
        except requests.RequestException as e:
            logger.error(f"Failed to fetch playlist: {e}")
            return None

    def download_segments(self, playlist: m3u8.M3U8, base_url: str) -> List[Path]:
        """Download all segments from the playlist."""
        segments: List[Path] = []
        for i, segment in enumerate(playlist.segments):
            segment_url = segment.absolute_uri if segment.absolute_uri.startswith(
                "http") else base_url + segment.uri
            segment_path = self.output_dir / f"segment_{i}.ts"
            try:
                response = self.session.get(segment_url)
                response.raise_for_status()
                with open(segment_path, "wb") as f:
                    f.write(response.content)
                segments.append(segment_path)
                logger.info(f"Downloaded segment: {segment_path}")
            except requests.RequestException as e:
                logger.error(f"Failed to download segment {segment_url}: {e}")
        return segments

    def download_subtitles(self, subtitle_url: str, output_file: str) -> bool:
        """Download subtitles from a given URL."""
        try:
            response = self.session.get(subtitle_url)
            response.raise_for_status()
            subtitle_path = self.output_dir / output_file
            with open(subtitle_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Downloaded subtitles to: {subtitle_path}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to download subtitles: {e}")
            return False

    def merge_segments(self, segments: List[Path], output_file: str, subtitle_file: Optional[str] = None) -> bool:
        """Merge downloaded segments into a single video file, optionally embedding subtitles."""
        output_path = self.output_dir / output_file
        try:
            input_files = [str(segment) for segment in segments]
            stream = ffmpeg.input("concat:" + "|".join(input_files))
            kwargs = {"c": "copy", "loglevel": "error"}
            if subtitle_file and (self.output_dir / subtitle_file).exists():
                kwargs["vf"] = f"subtitles={subtitle_file}"
            stream = ffmpeg.output(stream, str(output_path), **kwargs)
            ffmpeg.run(stream)
            logger.info(f"Merged video saved to: {output_path}")
            return True
        except ffmpeg.Error as e:
            logger.error(f"Failed to merge segments: {e}")
            return False

    def cleanup(self, segments: List[Path]) -> None:
        """Remove temporary segment files."""
        for segment in segments:
            try:
                segment.unlink()
                logger.info(f"Deleted segment: {segment}")
            except OSError as e:
                logger.error(f"Failed to delete segment {segment}: {e}")

    def download_video(self, m3u8_url: str, output_file: str, subtitle_url: Optional[str] = None) -> bool:
        """Download and merge a streaming video from an m3u8 URL."""
        if subtitle_url:
            self.download_subtitles(
                subtitle_url, f"{output_file.rsplit('.', 1)[0]}.srt")

        playlist = self.fetch_playlist(m3u8_url)
        if not playlist or not playlist.segments:
            logger.error("No valid segments found in playlist")
            return False

        base_url = m3u8_url[:m3u8_url.rfind("/") + 1]
        segments = self.download_segments(playlist, base_url)
        if not segments:
            logger.error("No segments downloaded")
            return False

        success = self.merge_segments(
            segments, output_file, f"{output_file.rsplit('.', 1)[0]}.srt" if subtitle_url else None)
        self.cleanup(segments)
        return success


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

    def get_episode_urls(self, series_url: str, timeout: int = 30) -> List[str]:
        """Scrape episode URLs from a series page."""
        if not self.driver:
            self.setup_driver()
        try:
            self.driver.get(series_url)
            logger.debug(f"Loaded series page: {series_url}")
            episode_elements = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, "a.episode-link, a.ssl-item.ep-item"))
            )
            episode_urls = [
                elem.get_attribute("href")
                for elem in episode_elements
                if elem.get_attribute("href") and "watch" in elem.get_attribute("href")
            ]
            logger.debug(
                f"Found {len(episode_urls)} episode URLs: {episode_urls}")
            return episode_urls
        except Exception as e:
            logger.error(f"Failed to scrape episode URLs: {e}")
            return []
        finally:
            self.close_driver()

    def get_stream_urls(self, video_url: str, timeout: int = 30, play_selectors: List[str] = [
        ".play-button", ".vjs-play-control", "button[title='Play']",
        "div.play-icon", "span.vjs-icon-play", "button[aria-label='Play']"
    ]) -> Dict[str, Optional[str]]:
        """Detect the m3u8 and subtitle URLs from a running video page."""
        if not self.driver:
            self.setup_driver()
        try:
            self.driver.get(video_url)
            logger.debug(f"Loaded video page: {video_url}")

            # Try to select a streaming server (common on AniWatch)
            try:
                server_button = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, ".server-item, .player-server, button[data-server]"))
                )
                ActionChains(self.driver).move_to_element(
                    server_button).click().perform()
                logger.debug("Clicked server selection button")
            except Exception as e:
                logger.debug(f"No server button found or not clickable: {e}")

            # Try to click a play button from the provided selectors
            for selector in play_selectors:
                try:
                    play_button = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    ActionChains(self.driver).move_to_element(
                        play_button).click().perform()
                    logger.debug(
                        f"Clicked play button with selector: {selector}")
                    break
                except Exception as e:
                    logger.debug(
                        f"No play button found for selector {selector}: {e}")

            # Wait for video element
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "video"))
            )
            logger.debug("Video element detected")

            m3u8_url = None
            subtitle_url = None
            logs = self.driver.get_log("performance")
            for entry in logs:
                log = json.loads(entry["message"])["message"]
                if log["method"] == "Network.responseReceived":
                    url = log["params"]["response"]["url"]
                    if url.endswith(".m3u8"):
                        m3u8_url = url
                        logger.info(f"Found m3u8 URL: {url}")
                    elif url.endswith((".srt", ".vtt")):
                        subtitle_url = url
                        logger.info(f"Found subtitle URL: {url}")

            if not m3u8_url:
                logger.error("No m3u8 URL found in network logs")
            return {"m3u8_url": m3u8_url, "subtitle_url": subtitle_url}
        except Exception as e:
            logger.error(f"Failed to detect stream URLs: {e}")
            return {"m3u8_url": None, "subtitle_url": None}
        finally:
            self.close_driver()

    def download_running_video(self, video_url: str, output_file: str) -> bool:
        """Automatically download a running video from a given URL."""
        # Check if URL is a series page by attempting to scrape episode URLs
        episode_urls = self.get_episode_urls(video_url)
        target_url = episode_urls[0] if episode_urls else video_url
        if episode_urls:
            logger.info(
                f"Series page detected; using first episode URL: {target_url}")

        stream_urls = self.get_stream_urls(target_url)
        m3u8_url = stream_urls["m3u8_url"]
        subtitle_url = stream_urls["subtitle_url"]

        if not m3u8_url:
            logger.error("Could not detect video stream URL")
            return False

        return self.downloader.download_video(m3u8_url, output_file, subtitle_url)

    def download_all_episodes(self, series_url: str, output_dir: str) -> List[bool]:
        episode_urls = self.get_episode_urls(series_url)
        results = []
        for i, url in enumerate(episode_urls, 1):
            output_file = f"episode_{i}.mp4"
            results.append(self.download_running_video(url, output_file))
        return results
