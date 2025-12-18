from typing import TypedDict


class SpeechSegment(TypedDict):
    idx: int
    start: float | int
    end: float | int
    prob: float
    duration: float