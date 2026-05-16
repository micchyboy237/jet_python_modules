import shutil
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.helpers.silence import SAMPLE_RATE
from jet.audio.speech.segment_store import SegmentStore
from jet.audio.speech.utils import display_segments
from jet.audio.speech_detector import record_from_mic
from jet.audio.speech_handlers.base import SpeechSegmentHandler
from jet.audio.speech_handlers.speech_events import SpeechSegmentEndEvent
from jet.audio.speech_handlers.subtitle_overlay_window import SubtitleOverlay
from jet.audio.speech_handlers.websocket_subtitle_sender import (
    WebsocketSubtitleSender,
)
from jet.file.utils import save_file
from jet.logger import logger
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Speech handlers
# ---------------------------------------------------------------------------


def dispatch_handlers(
    handlers: list[SpeechSegmentHandler],
    speech_seg: SpeechSegment,
    seg_audio_np: np.ndarray,
    seg_dir: Path,
    seg_number: int,
    sample_rate: int,
    started_at: datetime,
) -> None:
    """Fire on_segment_end on every registered handler. Errors are caught per-handler."""
    event = SpeechSegmentEndEvent(
        segment=speech_seg,
        segment_number=seg_number,
        audio_np=seg_audio_np,
        segment_dir=seg_dir,
        started_at=started_at,
        sample_rate=sample_rate,
    )
    for handler in handlers:
        try:
            handler.on_segment_end(event)
        except Exception as exc:
            logger.error(f"Handler {type(handler).__name__} failed: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main_live_speech_translation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    _sigint_timer = QTimer()
    _sigint_timer.start(200)
    _sigint_timer.timeout.connect(lambda: None)

    handlers: list[SpeechSegmentHandler] = [
        WebsocketSubtitleSender(global_srt_path=OUTPUT_DIR / "subtitles.srt"),
    ]
    ws_sender: WebsocketSubtitleSender = handlers[0]

    all_segments_path = OUTPUT_DIR / "all_segments.json"
    subtitles_path = OUTPUT_DIR / "subtitles.srt"
    segment_store = SegmentStore(OUTPUT_DIR / "segments")

    completed_segments: list[SpeechSegment] = []

    def _on_clear() -> None:
        completed_segments.clear()
        segment_store.reset()
        ws_sender.clear_queue()

    overlay = SubtitleOverlay.create_and_connect(
        ws_sender,
        extra_clear_paths=[all_segments_path, subtitles_path],
        on_clear=_on_clear,
    )

    _stop_recording = threading.Event()

    def _recording_loop() -> None:
        recording_started_at = datetime.now(timezone.utc)
        data_stream = record_from_mic(
            duration=None,
            trim_silent=False,
            quit_on_silence=False,
            verbose=False,
        )
        for speech_seg, seg_audio_np in data_stream:
            if _stop_recording.is_set():
                break

            # ── Guard: skip empty or near-silent segments ──────────────────────
            if seg_audio_np is None or seg_audio_np.size == 0:
                logger.warning(
                    f"[recorder] Skipping segment with empty audio "
                    f"(start={speech_seg.get('start'):.2f}s, "
                    f"reason={speech_seg.get('end_reason')})"
                )
                continue
            # ───────────────────────────────────────────────────────────────────

            seg_dir, seg_number = segment_store.save(
                speech_seg, seg_audio_np, sample_rate=SAMPLE_RATE
            )

            speech_seg["num"] = seg_number

            dispatch_handlers(
                handlers,
                speech_seg,
                seg_audio_np,
                seg_dir,
                seg_number,
                SAMPLE_RATE,
                recording_started_at,
            )

            _speech_seg_no_probs = speech_seg.copy()
            _speech_seg_probs = _speech_seg_no_probs.pop("segment_probs")

            completed_segments.append(_speech_seg_no_probs)

            save_file(_speech_seg_probs, seg_dir / "probs.json")
            save_file(completed_segments, all_segments_path)

        display_segments(completed_segments, done=True)

    rec_thread = threading.Thread(target=_recording_loop, daemon=True, name="recorder")
    rec_thread.start()

    def _shutdown() -> None:
        logger.info("[shutdown] Stopping recorder…")
        _stop_recording.set()
        logger.info("[shutdown] Closing WebSocket sender…")
        for h in handlers:
            if hasattr(h, "close"):
                h.close()
        logger.info("[shutdown] Joining recorder thread…")
        rec_thread.join(timeout=5.0)
        logger.info("[shutdown] Done.")

    app.aboutToQuit.connect(_shutdown)
    overlay.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main_live_speech_translation()
