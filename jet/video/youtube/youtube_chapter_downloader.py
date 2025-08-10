import subprocess
import os

from jet.video.utils import download_audio


class YoutubeChapterDownloader:
    def split_youtube_chapters(self, audio_dir, video_url, chapters):
        audio_format = "mp3"
        audio_path = download_audio(
            video_url, audio_dir, audio_format)

        if not audio_path:
            print("Failed to download audio.")
            return

        chapter_audio_items = []

        # Split the audio based on chapters
        for idx, chapter in enumerate(chapters):
            start = chapter['chapter_start']
            end = chapter['chapter_end']
            chapter_title = chapter['chapter_title'].replace(
                '/', '-').replace('|', '-')  # Sanitize title
            chapter_title = "_".join(chapter_title.split()).lower()
            chapter_file_path = os.path.join(
                audio_dir, f"chapters/{idx + 1}_{chapter_title}.{audio_format}")

            obj = {
                "chapter_title": chapter['chapter_title'],
                "chapter_start": start,
                "chapter_end": end,
                "chapter_file_path": chapter_file_path
            }

            if os.path.exists(chapter_file_path):
                print(f"Chapter {chapter_title} already exists.")
                chapter_audio_items.append(obj)
                continue

            os.makedirs(os.path.dirname(chapter_file_path), exist_ok=True)

            subprocess.run([
                'ffmpeg',
                '-i', audio_path,  # Input file
                '-ss', start,  # Start time
                '-to', end,  # End time
                '-c', 'copy',  # Copy audio without re-encoding
                chapter_file_path
            ])

            chapter_audio_items.append(obj)

        print("All chapters have been downloaded and split.")
        return chapter_audio_items

    def has_video(self, video_dir):
        video_path = f'{video_dir}/video.mp4'
        return os.path.exists(video_path)

    def has_audio(self, audio_dir):
        video_path = f'{audio_dir}/audio.mp3'
        return os.path.exists(video_path)
