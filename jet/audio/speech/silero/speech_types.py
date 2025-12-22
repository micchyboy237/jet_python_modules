from typing import TypedDict
from typing import List


class SpeechSegment(TypedDict):
    num: int
    start: float | int
    end: float | int
    prob: float
    duration: float
    frames_length: int
    frame_start: int
    frame_end: int
    segment_probs: List[float]  # Only present when with_scores=True