from typing import List, Optional, TypedDict


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
