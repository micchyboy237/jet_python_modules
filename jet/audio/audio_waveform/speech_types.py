from typing import Literal, TypedDict

VadStateLabel = Literal[
    "UNKNOWN", "SILENCE", "POSSIBLE_SPEECH", "SPEECH", "POSSIBLE_SILENCE"
]


class SpeechFrame(TypedDict):
    frame_idx: int
    raw_prob: float
    smoothed_prob: float
    is_speech: bool
    is_speech_start: bool
    is_speech_end: bool
    speech_start_frame: int
    speech_end_frame: int
    vad_state: VadStateLabel
