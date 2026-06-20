import shutil
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.helpers.silence import SAMPLE_RATE
from jet.audio.normalization.norm_speech_loudness import normalize_audio_for_vad
from jet.audio.speech.segment_store import SegmentStore
from jet.audio.speech.utils import display_segments
from jet.audio.speech_detector import record_from_mic
from jet.audio.speech_handlers.base import SpeechSegmentHandler
from jet.audio.speech_handlers.global_reset_handler import GlobalResetHandler
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
    # Normalize the audio before further processing
    seg_audio_np, _ = normalize_audio_for_vad(seg_audio_np, sample_rate)

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


def main_live_speech_translation(verbose: bool = False):
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
    global_reset_handler = GlobalResetHandler()

    all_segments_path = OUTPUT_DIR / "all_segments.json"
    subtitles_path = OUTPUT_DIR / "subtitles.srt"
    segment_store = SegmentStore(OUTPUT_DIR / "segments")

    completed_segments: list[SpeechSegment] = []
    audio_stats = {
        "total_segments": 0,
        "segments_with_overflow": 0,
        "empty_segments": 0,
    }

    def _on_clear() -> None:
        completed_segments.clear()
        segment_store.reset()
        ws_sender.clear_queue()
        audio_stats.update(
            {
                "total_segments": 0,
                "segments_with_overflow": 0,
                "empty_segments": 0,
            }
        )

    overlay = SubtitleOverlay.create_and_connect(
        ws_sender,
        extra_clear_paths=[all_segments_path, subtitles_path],
        on_clear=_on_clear,
        global_reset_handler=global_reset_handler,
    )

    _stop_recording = threading.Event()

    def _recording_loop() -> None:
        recording_started_at = datetime.now(timezone.utc)
        logger.info(
            f"[recorder] Audio recording started at {recording_started_at.isoformat()}"
        )

        data_stream = record_from_mic(
            duration=None,
            trim_silent=False,
            quit_on_silence=False,
            verbose=verbose,
        )

        for speech_seg, seg_audio_np in data_stream:
            audio_stats["total_segments"] += 1

            # Check for overflow flags in segment metadata
            had_overflow = speech_seg.get("had_overflow", False)
            if had_overflow:
                audio_stats["segments_with_overflow"] += 1
                logger.warning(
                    f"[recorder] Segment {speech_seg['num']} affected by audio overflow! "
                    f"({audio_stats['segments_with_overflow']}/{audio_stats['total_segments']} segments affected)"
                )

            logger.success(
                f"Speech {speech_seg['num']}: "
                f"{speech_seg['start_time_utc']} → {speech_seg['end_time_utc']} "
                f"[{'⚠ OVERFLOW' if had_overflow else '✓ clean'}]"
            )

            if _stop_recording.is_set():
                break

            if seg_audio_np is None or seg_audio_np.size == 0:
                audio_stats["empty_segments"] += 1
                logger.warning(
                    f"[recorder] Skipping empty segment "
                    f"(start={speech_seg.get('start', 'unknown'):.2f}s, "
                    f"reason={speech_seg.get('end_reason', 'unknown')})"
                )
                continue

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
            _speech_seg_probs = _speech_seg_no_probs.pop("segment_probs", None)
            completed_segments.append(_speech_seg_no_probs)
            save_file(completed_segments, all_segments_path)

        # Final statistics
        quality_ok = audio_stats["segments_with_overflow"] == 0
        logger.info(
            f"[recorder] Recording complete. Stats:\n"
            f"  Total segments: {audio_stats['total_segments']}\n"
            f"  With overflow: {audio_stats['segments_with_overflow']}\n"
            f"  Empty segments: {audio_stats['empty_segments']}\n"
            f"  Audio quality: {'⚠ DEGRADED' if not quality_ok else '✓ GOOD'}"
        )

        display_segments(completed_segments, done=True)

    rec_thread = threading.Thread(
        target=_recording_loop, daemon=True, name="audio-recorder"
    )
    rec_thread.start()
    logger.info(f"[main] Audio recorder thread started (tid={rec_thread.ident})")

    def _shutdown() -> None:
        logger.info("[shutdown] Stopping recorder…")
        _stop_recording.set()

        logger.info("[shutdown] Closing WebSocket sender…")
        for h in handlers:
            if hasattr(h, "close"):
                h.close()

        logger.info("[shutdown] Joining recorder thread…")
        rec_thread.join(timeout=5.0)
        if rec_thread.is_alive():
            logger.warning("[shutdown] Recorder thread did not exit cleanly!")

        # Print final quality report
        if audio_stats["segments_with_overflow"] > 0:
            logger.warning(
                f"[shutdown] ⚠ Audio quality issues detected: "
                f"{audio_stats['segments_with_overflow']} segments affected by overflow."
            )
        else:
            logger.info("[shutdown] ✓ Audio quality was good throughout recording")

        logger.info("[shutdown] Done.")

    app.aboutToQuit.connect(_shutdown)
    overlay.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live Speech Translation")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    main_live_speech_translation(verbose=args.verbose)
