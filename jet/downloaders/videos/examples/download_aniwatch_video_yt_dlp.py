import os
import yt_dlp

from jet.logger import logger


def download_aniwatch_video(url: str, output_path: str) -> bool:
    """Download an AniWatch video with subtitles using yt-dlp."""
    ydl_opts = {
        "outtmpl": f"{output_path}/%(title)s.%(ext)s",
        "format": "bestvideo[height<=1080]+bestaudio/best",
        "merge_output_format": "mp4",
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "writesubtitles": True,
        "subtitleslangs": ["en"],
        "subtitle": "--embed-subs"
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        logger.info(f"Video downloaded to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download with yt-dlp: {e}")
        return False


# Usage
if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "downloads", os.path.splitext(os.path.basename(__file__))[0])
    video_url = "https://aniwatchtv.to/watch/dealing-with-mikadono-sisters-is-a-breeze-19765"
    download_aniwatch_video(video_url, output_dir)
