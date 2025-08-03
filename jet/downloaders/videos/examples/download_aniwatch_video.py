# download_anime.py
from jet.downloaders.videos.aniwatch_downloader import VideoDownloader, SeleniumDownloader

if __name__ == "__main__":
    downloader = VideoDownloader()
    selenium_downloader = SeleniumDownloader(downloader)
    video_url = "https://aniwatchtv.to/watch/dealing-with-mikadono-sisters-is-a-breeze-19765"
    selenium_downloader.download_running_video(
        video_url, "mikadono_sisters.mp4")
