# test_jet.downloaders.video_downloader.py
import pytest
from pathlib import Path
from typing import List
from jet.downloaders.videos.video_downloader import VideoDownloader, SeleniumDownloader
import m3u8
import requests
import json
from unittest.mock import Mock, patch


@pytest.fixture
def downloader(tmp_path: Path) -> VideoDownloader:
    """Create a VideoDownloader instance with a temporary output directory."""
    return VideoDownloader(output_dir=str(tmp_path))


@pytest.fixture
def selenium_downloader(downloader: VideoDownloader) -> SeleniumDownloader:
    """Create a SeleniumDownloader instance."""
    return SeleniumDownloader(downloader)


@pytest.fixture(autouse=True)
def cleanup_files(tmp_path: Path):
    """Clean up files after each test."""
    yield
    for file in tmp_path.glob("*.ts"):
        file.unlink()
    for file in tmp_path.glob("*.mp4"):
        file.unlink()


def test_fetch_playlist_variant(downloader: VideoDownloader):
    """Test fetching a variant playlist and selecting the highest quality."""
    # Given: A master playlist with multiple variants
    master_content = """
    #EXTM3U
    #EXT-X-STREAM-INF:BANDWIDTH=1000000,RESOLUTION=480x360
    low.m3u8
    #EXT-X-STREAM-INF:BANDWIDTH=5000000,RESOLUTION=1920x1080
    high.m3u8
    """
    high_content = """
    #EXTM3U
    #EXTINF:10,
    segment1.ts
    """

    def mock_get(url, *args, **kwargs):
        class MockResponse:
            def __init__(self, content, status_code=200):
                self.content = content.encode()
                self.status_code = status_code

            def raise_for_status(self):
                if self.status_code != 200:
                    raise requests.HTTPError("Error")
        if url.endswith("high.m3u8"):
            return MockResponse(high_content)
        return MockResponse(master_content)

    with patch.object(downloader.session, "get", mock_get):
        # When: Fetching the playlist
        playlist = downloader.fetch_playlist("http://example.com/master.m3u8")

        # Then: Expect the high-quality playlist to be returned
        assert playlist is not None
        assert len(playlist.segments) == 1
        assert playlist.segments[0].uri == "segment1.ts"


@patch("jet.downloaders.video_downloader.webdriver.Chrome")
def test_get_m3u8_url_success(mock_chrome, selenium_downloader: SeleniumDownloader):
    """Test detecting an m3u8 URL from a video page."""
    # Given: A mock Chrome driver with network logs
    mock_driver = Mock()
    mock_chrome.return_value = mock_driver
    mock_driver.get_log.return_value = [
        {
            "message": json.dumps({
                "message": {
                    "method": "Network.responseReceived",
                    "params": {
                        "response": {"url": "http://example.com/playlist.m3u8"}
                    }
                }
            })
        }
    ]
    mock_driver.find_element.return_value = Mock()  # Mock video element

    # When: Detecting the m3u8 URL
    m3u8_url = selenium_downloader.get_m3u8_url("http://aniwatchtv.to/video")

    # Then: Expect the correct m3u8 URL
    expected = "http://example.com/playlist.m3u8"
    assert m3u8_url == expected
    mock_driver.quit.assert_called_once()


@patch("jet.downloaders.video_downloader.webdriver.Chrome")
def test_download_running_video_failure(mock_chrome, selenium_downloader: SeleniumDownloader):
    """Test downloading a running video with no m3u8 URL detected."""
    # Given: A mock Chrome driver with no m3u8 logs
    mock_driver = Mock()
    mock_chrome.return_value = mock_driver
    mock_driver.get_log.return_value = []
    mock_driver.find_element.return_value = Mock()

    # When: Attempting to download
    success = selenium_downloader.download_running_video(
        "http://aniwatchtv.to/video", "output.mp4")

    # Then: Expect the download to fail
    expected = False
    assert success == expected
    mock_driver.quit.assert_called_once()
