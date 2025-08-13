import json
import math
import os
import re
from typing import List, Tuple
from playwright.async_api import async_playwright
from datetime import timedelta
from tqdm import tqdm
from jet.data.utils import generate_hash
from jet.file.utils import save_data, save_file
from jet.models.model_registry.transformers.speech_to_text.whisper_model_registry import WhisperModelRegistry
from jet.video.utils import deduplicate_all_transcriptions, time_str_to_seconds
from jet.video.youtube.youtube_chapter_downloader import YoutubeChapterDownloader
from jet.video.youtube.youtube_types import YoutubeTranscription


class YoutubeInfoExtractor:
    def __init__(self, video_url):
        self.video_url = video_url

    async def extract_info(self):
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
                await page.goto(self.video_url, timeout=10000)
                await page.wait_for_selector(".ytd-watch-info-text", timeout=10000)

                chapters = []
                chapter_elements = await page.query_selector_all(".ytd-macro-markers-list-item")
                duration_element = await page.query_selector(".ytp-time-display .ytp-time-duration")
                end_str = await duration_element.inner_text()
                video_duration = parse_time(end_str.strip())

                for i, chapter in enumerate(chapter_elements):
                    parent_element = await chapter.query_selector("..")
                    title_element = await chapter.query_selector("h4")
                    if title_element:
                        title = await title_element.get_attribute('title')
                        time_element = await parent_element.query_selector("#time")
                        start_time_str = await time_element.inner_text()
                        start_time = parse_time(start_time_str.strip())
                        if i < len(chapter_elements) - 1:
                            next_chapter_element = chapter_elements[i + 1]
                            next_parent_element = await next_chapter_element.query_selector("..")
                            next_time_element = await next_parent_element.query_selector("#time")
                            next_start_time_str = await next_time_element.inner_text()
                            end_time = parse_time(
                                next_start_time_str.strip()) - 1
                        else:
                            end_time = video_duration
                        if title not in [c['chapter_title'] for c in chapters]:
                            chapters.append({
                                'chapter_start': str(timedelta(seconds=start_time)),
                                'chapter_end': str(timedelta(seconds=end_time)),
                                'chapter_title': title
                            })

                duration = str(timedelta(seconds=video_duration))

                video_meta = await page.query_selector("meta[content*='youtube.com/watch']")
                video_id = await video_meta.get_attribute('content') if video_meta else None

                video_title_element = await page.query_selector(".ytp-title")
                video_title = await video_title_element.inner_text() if video_title_element else None

                video_info_element = await page.query_selector(".ytd-watch-info-text")
                video_info_strings = await video_info_element.inner_text() if video_info_element else ""
                # Log raw string
                print(f"DEBUG: Raw video_info_strings: '{video_info_strings}'")

                # Extract view count, date posted, and trending description more robustly
                view_count, date_posted, trending_description = video_info_strings.split(
                    '  ')

                channel_name_element = await page.query_selector(".ytd-channel-name")
                channel_name = await channel_name_element.inner_text() if channel_name_element else None

                subscriber_count_element = await page.query_selector(".ytd-channel-name + .ytd-channel-statistics")
                subscriber_count = await subscriber_count_element.inner_text() if subscriber_count_element else None
                subscriber_count = parse_subscriber_count(
                    subscriber_count.strip()) if subscriber_count else 0

                return {
                    "id": video_id,
                    'title': video_title,
                    'duration': duration,
                    'channel_name': channel_name,
                    'subscriber_count': subscriber_count,
                    'view_count': view_count,
                    'trending_description': trending_description,
                    'chapters': chapters,
                    'date_posted': date_posted,
                }
            except Exception as e:
                print(f"Timeout or error extracting video info: {e}")
                return {}
            finally:
                await browser.close()


def parse_subscriber_count(subscriber_str):
    num_part = subscriber_str.split()[0].strip()
    multiplier = 1
    if 'K' in num_part:
        multiplier = 1000
        num_part = num_part.replace('K', '')
    elif 'M' in num_part:
        multiplier = 1000000
        num_part = num_part.replace('M', '')
    return int(float(num_part) * multiplier)


def parse_time(time_str):
    parts = time_str.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0


def get_chapter_title_by_start_and_end_time(chapters, start_time, end_time):
    for chapter in chapters:
        start_time = int(start_time)
        end_time = int(end_time)
        chapter_start = int(chapter['chapter_start'])
        chapter_end = int(chapter['chapter_end'])
        if end_time >= chapter_start and end_time <= chapter_end:
            return chapter['chapter_title']
    return None


def transcribe_youtube_video_info(video_id: str, transcriptions: List[YoutubeTranscription], output_dir: str) -> None:
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    extractor = YoutubeInfoExtractor(video_url)
    results = extractor.extract_info()
    channel_name = results.get('channel_name')
    chapters = results.get('chapters')
    model_size = 'small'
    model = WhisperModelRegistry.load_model(model_size)
    downloader = YoutubeChapterDownloader()
    if not chapters:
        raise Exception(
            "No chapters found in this video or failed to fetch video content.")
    channel_name = "_".join(channel_name.split()).lower()
    audio_dir = f'{output_dir}/{channel_name}/{video_id}'
    os.makedirs(audio_dir, exist_ok=True)
    info_file_name = f"{audio_dir}/info.json"
    transcriptions_file_path = f"{audio_dir}/transcriptions.json"
    transcriptions_info_file_path = f"{audio_dir}/transcriptions_info.json"
    if not os.path.exists(info_file_name):
        save_file(results, info_file_name)
        print(f"Video info saved to {info_file_name}")
    else:
        print(f"Info already exists at {info_file_name}")
    chapter_audio_items = downloader.split_youtube_chapters(
        audio_dir, video_url, chapters)
    audio_path = f"{audio_dir}/audio.mp3"
    transcription_segments, transcription_info = model.transcribe(audio_path)
    save_file(transcription_info, transcriptions_info_file_path)
    converted_chapters = []
    for chapter in chapter_audio_items:
        converted_chapters.append({
            "chapter_title": chapter['chapter_title'],
            "chapter_start": chapter['chapter_start'],
            "chapter_end": chapter['chapter_end'],
            "chapter_file_path": chapter['chapter_file_path']
        })

    for idx, segment in enumerate(tqdm(transcription_segments, desc="Transcribing segments", unit="segment")):
        tqdm_desc = f"Transcribing segment {idx+1}/{len(transcription_segments)}"
        tqdm.write(tqdm_desc)
        if not segment.text:
            continue
        chapter_title = get_chapter_title_by_start_and_end_time(
            converted_chapters, segment.start, segment.end)
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
            "chapter_title": chapter_title,
            "text": segment.text,
            "info": {
                "video_id": video_id,
                "channel_name": results['channel_name'],
                "video_title": results['title'],
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

    # deduplicate_all_transcriptions(transcriptions)


def transcribe_youtube_videos_info(video_ids: List[str], output_dir: str) -> List[YoutubeTranscription]:
    transcriptions: List[YoutubeTranscription] = []
    for video_id in video_ids:
        try:
            transcribe_youtube_video_info(video_id, transcriptions, output_dir)
        except Exception as e:
            print(f"Error transcribing video {video_id}\n{e}")
            continue
    return transcriptions
