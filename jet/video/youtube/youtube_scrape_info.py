import asyncio
import json
import math
import os
from typing import Generator, Iterable, List, Dict, Any, Optional, TypedDict

from tqdm import tqdm
from jet.data.utils import generate_hash
from jet.file.utils import save_file, save_data
from jet.logger import logger
from jet.models.model_registry.transformers.speech_to_text.whisper_model_registry import WhisperModelRegistry
from jet.video.utils import download_audio, deduplicate_all_transcriptions, time_str_to_seconds
from jet.video.youtube.youtube_info_extractor import (
    YoutubeInfoExtractor,
    parse_time,
    get_chapter_title_by_start_and_end_time,
)
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.utils import make_chunks
from jet.video.youtube.youtube_chapter_downloader import YoutubeChapterDownloader
from faster_whisper.transcribe import Segment, TranscriptionInfo

from jet.video.youtube.youtube_types import YoutubeMetadata


def transcribe_audio_chunk(chunk, model: WhisperModel, temp_file: str):
    chunk.export(temp_file, format="mp3")
    segments, info = model.transcribe(temp_file)
    return segments, info


def transcribe_in_segments(audio_path: str, metadata: YoutubeMetadata, output_dir: str, chunk_duration: int = 120, model_size: str = 'small') -> tuple:
    print(f"Loading whisper with model size: {model_size}")
    model = WhisperModelRegistry.load_model(model_size)
    audio = AudioSegment.from_file(audio_path)
    chunk_length_ms = chunk_duration * 1000
    print(f"Creating chunks of duration {chunk_duration}s each")
    chunks = make_chunks(audio, chunk_length_ms)
    transcription_segments = []
    transcription_info = []
    segment_idx = 0
    for i, chunk in enumerate(tqdm(chunks, desc="Transcribing chunks", unit="chunk")):
        chunk_num = i + 1
        tqdm_desc = f"Transcribing chunk {chunk_num}/{len(chunks)}"
        tqdm.write(tqdm_desc)
        temp_dir = os.path.dirname(audio_path)
        temp_file = f"{temp_dir}/temp_segment.mp3"
        logger.info(
            f"Creating and transcribing temp file {chunk_num}:\n{temp_file}")
        chunk_file_dir = f"{output_dir}/chunks"
        chunk_segments_file_path = f"{chunk_file_dir}/chunk_{chunk_num}_segments.json"
        chunk_info_file_path = f"{chunk_file_dir}/chunk_{chunk_num}_info.json"

        segments, info = transcribe_audio_chunk(chunk, model, temp_file)

        save_file(info, chunk_info_file_path)

        processed_segments_stream = process_segments(segments, metadata)
        processed_segments = []
        for processed_segment in processed_segments_stream:
            chunk_data = {
                "segment_idx": segment_idx,
                **processed_segment,
            }
            processed_segments.append(chunk_data)
            save_file({
                "chunk_num": chunk_num,
                "segments_count": len(processed_segments),
                "segments": processed_segments
            }, chunk_segments_file_path)

            transcription_segments.append(chunk_data)

            segment_idx += 1
        transcription_info.append(info)
    return transcription_segments, transcription_info


def process_segments(segments: Generator[Segment, None, None], metadata: YoutubeMetadata) -> Generator:
    for idx, segment in enumerate(segments):
        logger.teal(f"Transcribing segment {idx+1}")
        if not segment.text:
            continue
        chapter_title = get_chapter_title_by_start_and_end_time(
            metadata["chapters"], segment.start, segment.end) if metadata["chapters"] else None
        # Convert avg_logprob from log space to probability
        confidence = round(math.exp(segment.avg_logprob), 4)
        transcription = {
            "id": generate_hash({
                "video_id": metadata["video_id"],
                "start": segment.start,
                "end": segment.end,
            }),
            "seek": segment.seek,
            "start": segment.start,
            "end": segment.end,
            "chapter_title": chapter_title,
            "text": segment.text,
            "eval": {
                "confidence": confidence,
                "temperature": segment.temperature,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
            },
            "words": segment.words,
        }
        yield transcription


def extract_metadata(info: Dict[str, Any], audio_dir: str, video_url: str) -> YoutubeMetadata:
    chapters = info.get('chapters', [])
    video_id = info.get('id', '')
    channel_name = "_".join(info.get('channel_name', '').split()).lower()
    subscriber_count = info.get('subscriber_count', 0)
    video_title = info.get('title', '')
    description = info.get('description', '')
    upload_date = info.get('date_posted', '')
    duration = info.get('duration', '')
    view_count = info.get('view_count', '')
    trending_description = info.get('trending_description', '')

    downloader = YoutubeChapterDownloader()
    chapter_audio_items = downloader.split_youtube_chapters(
        audio_dir, video_url, chapters) if chapters else []

    return {
        "video_id": video_id,
        "channel_name": channel_name,
        "subscriber_count": subscriber_count,
        "video_title": video_title,
        "video_url": video_url,
        "description": description,
        "upload_date": upload_date,
        "duration": duration,
        "view_count": view_count,
        "trending_description": trending_description,
        "chapters": chapter_audio_items,
    }


def transcribe_youtube_video_info_and_chapters(video_url: str, audio_path: str, info: Dict[str, Any], output_dir: str, chunk_duration: int = 120, model_size: str = 'small') -> None:
    audio_dir = output_dir
    os.makedirs(audio_dir, exist_ok=True)
    transcription_segments_file_path = f"{audio_dir}/transcription_segments.json"
    transcriptions_info_file_path = f"{audio_dir}/transcriptions_info.json"

    metadata = extract_metadata(info, audio_dir, video_url)

    transcription_segments, transcription_info = transcribe_in_segments(
        audio_path, metadata, output_dir, chunk_duration=chunk_duration, model_size=model_size)

    save_file(transcription_segments, transcription_segments_file_path)
    save_file(transcription_info, transcriptions_info_file_path)


def find_audio(audio_dir: str) -> list:
    mp3_files = [os.path.join(audio_dir, file) for file in os.listdir(
        audio_dir) if file.endswith('.mp3')]
    return mp3_files
