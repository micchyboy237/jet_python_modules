import json
import math
import os
from typing import Iterable
from playwright.async_api import async_playwright
from urllib.parse import urlparse, parse_qs

from tqdm import tqdm
from jet.data.utils import generate_hash
from jet.file.utils import save_data, save_file
from jet.video.utils import download_audio
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment


class YoutubePlaylistExtractor:
    def __init__(self, playlist_url):
        self.playlist_url = playlist_url

    async def extract_video_ids(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-infobars",
                    "--disable-notifications",
                    "--disable-gpu",
                    "--disable-extensions",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--incognito",
                    "--mute-audio"
                ]
            )
            try:
                context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                page = await context.new_page()
                await page.goto(self.playlist_url, timeout=10000)
                await page.wait_for_selector('#video-title', timeout=10000)

                video_ids = set()
                anchor_elements = await page.query_selector_all('#video-title')
                for element in anchor_elements:
                    url = await element.get_attribute('href')
                    if url:
                        query_string = urlparse(url).query
                        params = parse_qs(query_string)
                        video_id = params.get('v', [None])[0]
                        if video_id:
                            video_ids.add(video_id)
                return list(video_ids)
            except Exception as e:
                print(f"Error extracting video IDs: {e}")
                return []
            finally:
                await browser.close()


def has_video(video_dir):
    video_path = f'{video_dir}/video.mp4'
    return os.path.exists(video_path)


def has_audio(audio_dir):
    video_path = f'{audio_dir}/audio.mp3'
    return os.path.exists(video_path)


def download_youtube_audio(video_id, audio_dir):
    audio_format = "mp3"
    audio_path = download_audio(
        f"https://www.youtube.com/watch?v={video_id}", audio_dir, audio_format)
    return audio_path


def transcribe_youtube_video_playlist(model: WhisperModel, video_id, audio_dir, playlist_title):
    audio_path = f'{audio_dir}/audio.mp3'
    transcriptions_file_path = f"{audio_dir}/transcriptions.json"
    transcriptions_info_file_path = f"{audio_dir}/transcriptions_info.json"

    transcription_segments, transcription_info = model.transcribe(audio_path)

    save_file(transcription_info, transcriptions_info_file_path)

    transcriptions = []
    for idx, segment in enumerate(tqdm(transcription_segments, desc="Transcribing segments", unit="segment")):
        tqdm_desc = f"Transcribing segment {idx+1}/{len(transcription_segments)}"
        tqdm.write(tqdm_desc)
        if not segment.text:
            continue
        # Convert avg_logprob from log space to probability
        confidence = round(math.exp(segment.avg_logprob), 4)
        transcription = {
            "id": generate_hash({
                "video_id": video_id,
                "start": segment.start,
                "end": segment.end,
            }),
            "seek": segment.seek,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "info": {
                "video_id": video_id,
                "playlist_title": playlist_title,
            },
            "eval": {
                "confidence": confidence,
                "temperature": segment.temperature,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
            },
            "words": segment.words,
        }
        transcriptions.append(transcription)
        save_file(transcriptions, transcriptions_file_path)


def transcribe_youtube_videos_playlist(model, video_ids, playlist_title, output_dir: str):
    audio_playlist_dir = f'{output_dir}/{playlist_title}'

    for video_id in video_ids:
        audio_dir = f'{audio_playlist_dir}/{video_id}'
        if not has_audio(audio_dir):
            download_youtube_audio(video_id, audio_dir)
        transcribe_youtube_video_playlist(
            model, video_id, audio_dir, playlist_title)
