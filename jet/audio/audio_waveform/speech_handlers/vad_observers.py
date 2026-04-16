from pathlib import Path
from typing import Protocol

import numpy as np
from jet.audio.audio_waveform.helpers.subtitle_entry import SubtitleEntry
from jet.audio.audio_waveform.speech_handlers.live_srt_preview_handler import (
    LiveSrtPreviewHandler,
)
from jet.audio.audio_waveform.speech_handlers.speech_segment_saver import (
    SpeechSegmentSaver,
)
from jet.audio.audio_waveform.speech_handlers.websocket_subtitle_sender import (
    WebsocketSubtitleSender,
)
from jet.audio.audio_waveform.speech_segment_tracker import SpeechSegmentTracker
from jet.audio.audio_waveform.vad.firered_with_speech_tracking import FireRedVADWrapper
from jet.audio.audio_waveform.vad.silero import SileroVAD
from jet.audio.audio_waveform.vad.speechbrain import SpeechBrainVADWrapper
from jet.audio.audio_waveform.vad.ten_vad import TenVadWrapper  # NEW
from jet.audio.helpers.energy_base import compute_rms, normalize_energy


class AudioObserver(Protocol):
    def __call__(self, samples: np.ndarray) -> None: ...


class WaveformRMSObserver:
    """Observer that calculates normalized RMS energy for visualization.

    Uses a rolling window of recent RMS values to provide stable normalization
    via the normalize_energy function with adaptive max_rms.
    """

    def __init__(self):
        self.value: float = 0.0  # Normalized RMS value [0, 1]
        self.raw_rms: float = 0.0  # Raw RMS for debugging
        self._rms_history: list = []  # Rolling window for normalization
        self._history_size: int = 10  # Keep last 10 values for max calculation

    def __call__(self, samples: np.ndarray):
        if samples.size == 0:
            self.value = 0.0
            self.raw_rms = 0.0
            return

        # Calculate raw RMS
        self.raw_rms = compute_rms(samples)

        # Update history and normalize
        self._rms_history.append(self.raw_rms)
        if len(self._rms_history) > self._history_size:
            self._rms_history.pop(0)

        # Use max from recent history for stable normalization
        max_rms = max(self._rms_history) if self._rms_history else self.raw_rms
        self.value = float(normalize_energy([self.raw_rms], max_rms=max_rms)[0])


class VADObserver:
    def __init__(self, model):
        self.model = model
        self.probability: float = 0.0

    def __call__(self, samples: np.ndarray):
        self.probability = self.model.get_speech_prob(samples) if self.model else 0.0


class TrackerObserver:
    def __init__(self, tracker: SpeechSegmentTracker):
        self.tracker = tracker

    def __call__(self, samples: np.ndarray):
        if self.tracker:
            self.tracker.add_audio(samples)


def create_original_observers(
    samplerate: int,
    tracker_params: dict,
    firered_params: dict,
    output_dir: Path,
) -> dict:
    """
    Instantiates all VAD models and trackers exactly as they
    were configured in the original monolithic application.

    Returns:
        A dictionary of ready-to-use Observer instances.
    """

    # 1. Initialize the shared Speech Tracker
    # This maintains the state of speech segments (start/end times)
    tracker = SpeechSegmentTracker(
        speech_threshold=tracker_params.get("speech_threshold", 0.3)
    )

    SAVED_SPEECH_SEGMENTS_DIR = output_dir / "saved_speech_segments"
    SAVED_SPEECH_SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_SRT_PATH = output_dir / "live_subtitles.srt"

    saver = SpeechSegmentSaver(base_save_dir=SAVED_SPEECH_SEGMENTS_DIR)
    subtitle_entries = SubtitleEntry(output_path=GLOBAL_SRT_PATH)

    # WebSocket sender (binary + UUID matching)
    ws_handler = WebsocketSubtitleSender(
        accumulator=subtitle_entries,
        debug_save_audio=True,
    )

    # New: real-time preview handler
    preview_handler = LiveSrtPreviewHandler(accumulator=subtitle_entries)

    # Register handlers
    tracker.add_handler(saver)
    tracker.add_handler(ws_handler)
    tracker.add_handler(preview_handler)

    # 2. Initialize VAD Models
    # Silero: Fast, local ONNX-based VAD
    silero_model = SileroVAD(samplerate=samplerate)

    # SpeechBrain: Robust research-grade VAD
    sb_model = SpeechBrainVADWrapper()

    # FireRed: Advanced VAD that requires the tracker to manage segments
    fr_model = FireRedVADWrapper(tracker=tracker, **firered_params)

    # NEW: TEN-VAD with wrapper for compatibility
    ten_vad_model = TenVadWrapper(hop_size=160, threshold=0.5)

    # 3. Wrap everything in Observer classes
    # These classes provide a standard __call__(samples) interface
    observers = {
        "waveform": WaveformRMSObserver(),
        "silero": VADObserver(silero_model),
        "speechbrain": VADObserver(sb_model),
        "firered": VADObserver(fr_model),
        "ten_vad": VADObserver(ten_vad_model),  # NEW
        "tracker": TrackerObserver(tracker),
    }

    return observers
