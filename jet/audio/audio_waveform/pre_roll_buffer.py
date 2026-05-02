from collections import deque
from typing import List, Tuple

import numpy as np
from jet.audio.audio_waveform.speech_types import SpeechFrame
from jet.audio.helpers.energy import has_sound


class PreRollBuffer:
    """Holds recent audio + probability frames before speech is detected.
    This is the 'pre-roll' that lets us capture the start of speech cleanly."""

    def __init__(self, maxlen: int = 200):
        self.audio: deque[np.ndarray] = deque(maxlen=maxlen)
        self.probs: deque[float] = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def add_audio(self, samples: np.ndarray) -> None:
        """Store raw audio chunk while not yet in a speech segment."""
        if len(samples) == 0:
            return
        self.audio.append(samples.astype(np.float32).copy())

    def add_prob(self, smoothed_prob: float) -> None:
        """Store VAD probability while not yet in a speech segment."""
        self.probs.append(float(smoothed_prob))

    def get_rising_edge(
        self, pre_prob_thres: float = 0.1
    ) -> Tuple[List[np.ndarray], List[dict], int]:
        """Return audio chunks + frame dicts starting from the last silence/low-prob point."""
        if len(self.audio) == 0 or len(self.probs) == 0:
            return [], [], 0

        audio_list = list(self.audio)
        prob_list = list(self.probs)

        # Last silent audio block
        last_silent_audio_idx = -1
        for i in range(len(audio_list)):
            if not has_sound(audio_list[i]):  # uses your existing helper
                last_silent_audio_idx = i

        # Last low probability frame
        last_low_prob_idx = -1
        for i in range(len(prob_list)):
            if prob_list[i] < pre_prob_thres:
                last_low_prob_idx = i

        start_idx_candidates = []
        if last_silent_audio_idx != -1:
            start_idx_candidates.append(last_silent_audio_idx + 1)
        if last_low_prob_idx != -1:
            start_idx_candidates.append(last_low_prob_idx + 1)

        if not start_idx_candidates:
            return [], [], 0

        rise_start_idx = max(start_idx_candidates)
        if rise_start_idx >= len(audio_list):
            return [], [], 0

        rise_audio = audio_list[rise_start_idx:]
        rise_probs = prob_list[rise_start_idx:]

        # Build frame entries for the prepended part
        rise_frames: list[SpeechFrame] = []
        base_frame_idx = 0  # Will be adjusted by caller if needed
        for i, prob in enumerate(rise_probs):
            rise_frames.append(
                {
                    "frame_idx": base_frame_idx + i,
                    "raw_prob": prob,
                    "smoothed_prob": prob,
                    "is_speech": prob >= pre_prob_thres,
                    "is_speech_start": False,
                    "is_speech_end": False,
                    "vad_state": "SPEECH",
                }
            )

        return rise_audio, rise_frames, rise_start_idx

    def clear(self) -> None:
        self.audio.clear()
        self.probs.clear()
