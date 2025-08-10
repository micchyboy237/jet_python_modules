import asyncio
import json
import os
from typing import List, Dict, Any
from jet.data.utils import generate_hash
from jet.file.utils import save_file, save_data
from jet.models.model_registry.transformers.speech_to_text.whisper_model_registry import WhisperModelRegistry
from jet.video.utils import download_audio, deduplicate_all_transcriptions
from jet.video.youtube.youtube_info_extractor import (
    YoutubeInfoExtractor,
    parse_time,
    time_str_to_seconds,
    get_chapter_title_by_start_and_end_time,
)
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.utils import make_chunks
from jet.video.youtube.youtube_chapter_downloader import YoutubeChapterDownloader


def transcribe_audio_chunk(chunk, model: WhisperModel, temp_dir: str):
    temp_file = f"{temp_dir}/temp_segment.mp3"
    print(f"Creating temp file:\n{temp_file}")
    chunk.export(temp_file, format="mp3")
    print(f"Transcribing temp file")
    segments, info = model.transcribe(temp_file)
    return segments, info


def transcribe_in_segments(audio_path: str, chunk_duration: int = 300, model_size: str = 'small') -> tuple:
    print(f"Loading whisper with model size: {model_size}")
    model = WhisperModelRegistry.load_model(model_size)
    audio = AudioSegment.from_file(audio_path)
    chunk_length_ms = chunk_duration * 1000
    print(f"Creating chunks of duration {chunk_duration}s each")
    chunks = make_chunks(audio, chunk_length_ms)
    transcription_segments = []
    transcription_info = []
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i}")
        temp_dir = os.path.dirname(audio_path)
        segments, info = transcribe_audio_chunk(chunk, model, temp_dir)
        for segment in segments:
            transcription_segments.append(segment)
        transcription_info.append(info)
    return transcription_segments, transcription_info


def transcribe_youtube_video_info_and_chapters(video_url: str, audio_path: str, info: Dict[str, Any], output_dir: str) -> List[Dict[str, Any]]:
    audio_dir = output_dir
    os.makedirs(audio_dir, exist_ok=True)
    transcriptions_file_path = f"{audio_dir}/transcriptions.json"
    transcriptions_info_file_path = f"{audio_dir}/transcriptions_info.json"

    chapters = info.get('chapters', [])
    video_id = info.get('id', '')
    channel_name = "_".join(info.get('channel_name', '').split()).lower()

    downloader = YoutubeChapterDownloader()
    chapter_audio_items = downloader.split_youtube_chapters(
        audio_dir, video_url, chapters) if chapters else []

    transcriptions: List[Dict[str, Any]] = []
    transcription_segments, transcription_info = transcribe_in_segments(
        audio_path)

    save_file(transcription_info, transcriptions_info_file_path)
    print(f"Transcription info saved to {transcriptions_info_file_path}")

    converted_chapters = [
        {
            "chapter_title": chapter['chapter_title'],
            "chapter_start": time_str_to_seconds(chapter['chapter_start']),
            "chapter_end": time_str_to_seconds(chapter['chapter_end']),
            "chapter_file_path": chapter['chapter_file_path']
        }
        for chapter in chapter_audio_items
    ]

    batch_items = []
    batch_size = 2
    for segment in transcription_segments:
        if not segment.text:
            continue
        chapter_title = get_chapter_title_by_start_and_end_time(
            converted_chapters, segment.start, segment.end) if chapters else None
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
                "channel_name": info.get('channel_name', ''),
                "video_title": info.get('title', ''),
            },
            "eval": {
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
            },
            "words": segment.words,
        }
        transcriptions.append(transcription)
        batch_items.append(transcription)
        if len(batch_items) >= batch_size:
            save_data(transcriptions_file_path, batch_items)
            batch_items = []

    if batch_items:
        save_data(transcriptions_file_path, batch_items)

    deduplicate_all_transcriptions(transcriptions)
    return transcriptions


def find_audio(audio_dir: str) -> list:
    mp3_files = [os.path.join(audio_dir, file) for file in os.listdir(
        audio_dir) if file.endswith('.mp3')]
    return mp3_files
