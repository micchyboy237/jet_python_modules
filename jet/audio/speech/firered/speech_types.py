from typing import List, Literal, Optional, TypedDict


class SpeechSegment(TypedDict):
    num: int
    start: float | int
    end: float | int
    prob: float
    duration: float
    frames_length: int
    frame_start: int
    frame_end: int
    type: Literal["speech", "non-speech"]
    segment_probs: List[float]


class WordSegment(TypedDict):
    index: int
    start_ms: Optional[int]
    end_ms: Optional[int]
    word: Optional[str]
