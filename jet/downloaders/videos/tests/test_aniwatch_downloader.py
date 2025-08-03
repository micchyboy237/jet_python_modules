import pytest
from pathlib import Path
from typing import List, Dict
from jet.downloaders.videos.aniwatch_downloader import VideoDownloader, SeleniumDownloader
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
    for file in tmp_path.glob("*.srt"):
        file.unlink()


def test_download_subtitles_success(downloader: VideoDownloader, tmp_path: Path):
    """Test downloading subtitles from a valid URL."""
    # Given: A mock subtitle URL
    subtitle_url = "http://example.com/subtitles.srt"
    output_file = "subtitles.srt"

    def mock_get(*args, **kwargs):
        class MockResponse:
            def __init__(self):
                self.content = b"1\n00:00:01,000 --> 00:00:02,000\nHello!"
                self.status_code = 200

            def raise_for_status(self):
                pass
        return MockResponse()

    downloader.session.get = mock_get

    # When: Downloading subtitles
    success = downloader.download_subtitles(subtitle_url, output_file)

    # Then: Expect subtitles to be downloaded correctly
    expected = tmp_path / output_file
    assert success
    assert expected.exists()
    assert expected.read_bytes() == b"1\n00:00:01,000 --> 00:00:02,000\nHello!"


# test_aniwatch_downloader.py
def test_fetch_playlist_variant(downloader: VideoDownloader):
    """Test fetching a variant playlist and selecting the highest quality."""
    # Given: A master playlist with multiple variants
    master_content = """
    #EXTM3U
    #EXT-X-STREAM-INF:BANDWIDTH=1000000,RESOLUTION=1280x720
    720p.m3u8
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
                self.text = content  # Add text property to decode content

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
        expected = m3u8.M3U8()
        expected.segments.append(m3u8.Segment(uri="segment1.ts", duration=10))
        assert playlist is not None
        assert len(playlist.segments) == len(expected.segments)
        assert playlist.segments[0].uri == expected.segments[0].uri


@patch("jet.downloaders.videos.aniwatch_downloader.webdriver.Chrome")
def test_get_stream_urls_success(mock_chrome, selenium_downloader: SeleniumDownloader):
    """Test detecting m3u8 and subtitle URLs from a video page with custom play selectors."""
    # Given: A mock Chrome driver with network logs and a video element
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
        },
        {
            "message": json.dumps({
                "message": {
                    "method": "Network.responseReceived",
                    "params": {
                        "response": {"url": "http://example.com/subtitles.srt"}
                    }
                }
            })
        }
    ]
    mock_driver.find_element.side_effect = [
        Mock(),  # Video element
        Exception("No server button"),  # No server button
        Mock()  # Play button for .vjs-play-control
    ]

    # When: Detecting stream URLs with custom selectors
    stream_urls = selenium_downloader.get_stream_urls(
        "https://aniwatchtv.to/watch/dealing-with-mikadono-sisters-is-a-breeze-19765",
        play_selectors=[".vjs-play-control", "button[title='Play']"]
    )

    # Then: Expect both m3u8 and subtitle URLs
    expected = {
        "m3u8_url": "http://example.com/playlist.m3u8",
        "subtitle_url": "http://example.com/subtitles.srt"
    }
    assert stream_urls == expected
    mock_driver.quit.assert_called_once()


@patch("jet.downloaders.videos.video_downloader.webdriver.Chrome")
def test_download_running_video_no_m3u8(mock_chrome, selenium_downloader: SeleniumDownloader):
    """Test downloading a running video with no m3u8 URL detected."""
    # Given: A mock Chrome driver with no m3u8 logs
    mock_driver = Mock()
    mock_chrome.return_value = mock_driver
    mock_driver.get_log.return_value = []
    mock_driver.find_element.return_value = Mock()

    # When: Attempting to download
    success = selenium_downloader.download_running_video(
        "https://aniwatchtv.to/watch/dealing-with-mikadono-sisters-is-a-breeze-19765",
        "output.mp4"
    )

    # Then: Expect the download to fail
    expected = False
    assert success == expected
    mock_driver.quit.assert_called_once()


@patch("jet.downloaders.videos.aniwatch_downloader.webdriver.Chrome")
def test_get_stream_urls_no_play_button(mock_chrome, selenium_downloader: SeleniumDownloader):
    """Test detecting stream URLs when no play button is found but video loads."""
    # Given: A mock Chrome driver with network logs and a video element
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
    mock_driver.find_element.side_effect = [
        Mock(),  # Video element
        Exception("No server button"),  # No server button
        Exception("No play button")  # No play button for any selector
    ]

    # When: Detecting stream URLs with custom selectors
    stream_urls = selenium_downloader.get_stream_urls(
        "https://aniwatchtv.to/watch/dealing-with-mikadono-sisters-is-a-breeze-19765",
        play_selectors=[".invalid-selector"]
    )

    # Then: Expect m3u8 URL despite no play button
    expected = {
        "m3u8_url": "http://example.com/playlist.m3u8",
        "subtitle_url": None
    }
    assert stream_urls == expected
    mock_driver.quit.assert_called_once()
