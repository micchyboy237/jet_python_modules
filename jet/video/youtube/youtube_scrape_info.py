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
from jet.video.utils import deduplicate_segments, download_audio, make_chunks_with_overlap, time_str_to_seconds
from jet.video.youtube.youtube_info_extractor import (
    YoutubeInfoExtractor,
    parse_time,
    get_chapter_title_by_start_and_end_time,
)
from faster_whisper import WhisperModel
from pydub import AudioSegment
from jet.video.youtube.youtube_chapter_downloader import YoutubeChapterDownloader
from faster_whisper.transcribe import Segment, TranscriptionInfo

from jet.video.youtube.youtube_types import YoutubeMetadata


def transcribe_audio_chunk(chunk, model: WhisperModel, temp_file: str):
    chunk.export(temp_file, format="mp3")
    segments, info = model.transcribe(temp_file, word_timestamps=True)
    return segments, info


def process_segments(
    segments: Generator[Segment, None, None],
    metadata: YoutubeMetadata,
    chunk_start_ms: float,
) -> Generator:
    for idx, segment in enumerate(segments):
        logger.teal(f"Transcribing segment {idx+1}")
        if not segment.text:
            continue
        chapter_title = get_chapter_title_by_start_and_end_time(
            metadata["chapters"], segment.start + chunk_start_ms /
            1000, segment.end + chunk_start_ms / 1000
        ) if metadata["chapters"] else None
        confidence = round(math.exp(segment.avg_logprob), 4)
        transcription = {
            "id": generate_hash({
                "video_id": metadata["video_id"],
                "start": segment.start + chunk_start_ms / 1000,
                "end": segment.end + chunk_start_ms / 1000,
            }),
            # seek is in 10ms units
            "seek": segment.seek + int(chunk_start_ms / 10),
            "start": segment.start + chunk_start_ms / 1000,
            "end": segment.end + chunk_start_ms / 1000,
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
        logger.debug(
            f"Segment {idx+1}: original start={segment.start}s, end={segment.end}s, "
            f"adjusted start={transcription['start']}s, end={transcription['end']}s, "
            f"chunk_start_ms={chunk_start_ms}ms"
        )
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


def transcribe_youtube_video_info_and_chapters(
    video_url: str,
    audio_path: str,
    info: Dict[str, Any],
    output_dir: str,
    chunk_duration: int = 120,
    model_size: str = "small",
    overlap_duration: float = 0.0,
) -> Generator[tuple[list, Any], None, None]:
    audio_dir = output_dir
    os.makedirs(audio_dir, exist_ok=True)
    metadata = extract_metadata(info, audio_dir, video_url)
    print(f"Loading whisper with model size: {model_size}")
    model = WhisperModelRegistry.load_model(model_size)
    audio = AudioSegment.from_file(audio_path)
    chunk_length_ms = chunk_duration * 1000
    overlap_ms = int(overlap_duration * 1000)
    print(
        f"Creating chunks of duration {chunk_duration}s with {overlap_duration}s overlap")
    chunks = make_chunks_with_overlap(
        audio, chunk_length_ms, overlap_ms=overlap_ms)
    segment_idx = 0
    total_duration = len(audio) / 1000
    all_segments = []
    with tqdm(total=len(chunks), desc="Transcribing audio chunks", unit="chunk") as pbar:
        for i, chunk in enumerate(chunks):
            chunk_num = i + 1
            chunk_start_ms = i * (chunk_duration - overlap_duration) * 1000
            chunk_end = min(chunk_start_ms / 1000 +
                            chunk_duration, total_duration)
            tqdm_desc = f"Chunk {chunk_num}/{len(chunks)} ({chunk_start_ms/1000:.1f}s - {chunk_end:.1f}s)"
            pbar.set_description(tqdm_desc)
            temp_dir = os.path.dirname(audio_path)
            temp_file = f"{temp_dir}/temp_segment.mp3"
            logger.info(f"Transcribing chunk {chunk_num}: {temp_file}")
            logger.debug(
                f"Chunk {chunk_num}: start={chunk_start_ms/1000}s, end={chunk_end}s, "
                f"duration={len(chunk)/1000}s"
            )
            chunk_file_dir = f"{audio_dir}/chunks"
            os.makedirs(chunk_file_dir, exist_ok=True)
            segments, info = transcribe_audio_chunk(chunk, model, temp_file)
            processed_segments = []
            for processed_segment in process_segments(segments, metadata, chunk_start_ms):
                processed_segments.append(
                    processed_segment)  # No segment_idx yet
                all_segments.append(processed_segment)
            deduplicated_segments = deduplicate_segments(all_segments)
            # Assign segment_idx to deduplicated segments
            indexed_segments = [
                {"chunk_idx": i, "segment_idx": segment_idx + j, **segment}
                for j, segment in enumerate(deduplicated_segments)
            ]
            logger.debug(
                f"Chunk {chunk_num}: {len(processed_segments)} segments before deduplication, "
                f"{len(deduplicated_segments)} segments after deduplication, "
                f"assigned indices {segment_idx} to {segment_idx + len(deduplicated_segments) - 1}"
            )
            yield indexed_segments, info
            all_segments = deduplicated_segments
            segment_idx += len(deduplicated_segments)
            pbar.update(1)
