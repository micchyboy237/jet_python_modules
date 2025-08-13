from typing import List, Optional, TypedDict
from faster_whisper.transcribe import Segment, TranscriptionInfo


class YoutubeTranscription(TypedDict):
    id: str
    seek: int
    start: float
    end: float
    chapter_title: Optional[str]
    text: str
    info: dict
    eval: dict
    words: list


class YoutubeChapter(TypedDict):
    chapter_title: str
    chapter_start: int
    chapter_end: int
    chapter_file_path: str


class YoutubeMetadata(TypedDict):
    video_id: str
    channel_name: str
    subscriber_count: int
    video_title: str
    video_url: str
    description: str
    upload_date: str
    duration: str
    view_count: str
    trending_description: str
    chapters: list[YoutubeChapter]


class TranscriptionChunk(TypedDict):
    chunk_num: int
    segments_count: int
    segments: List[Segment]
    info: TranscriptionInfo
