from typing import TypedDict
from typing import List

class SpeechWaveMeta(TypedDict):
    has_risen: bool
    has_multi_passed: bool
    has_fallen: bool
    is_valid: bool

class SpeechWave(SpeechWaveMeta):
    start_sec: float
    end_sec: float

class SpeechSegment(TypedDict):
    num: int
    start: float | int
    end: float | int
    prob: float
    duration: float
    frames_length: int
    frame_start: int
    frame_end: int
    # Only present when with_scores=True
    segment_probs: List[float]  
    # speech_waves: List[SpeechWave]