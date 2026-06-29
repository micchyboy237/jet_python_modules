import shutil
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.audio_waveform.vad.vad_config import (
    DEFAULT_ACC_MAX_DURATION_SEC,
    DEFAULT_MAX_SEG_DURATION_SEC,
    DEFAULT_MAX_SEG_GAP_SEC,
    DEFAULT_MIN_SEG_DURATION_SEC,
    DEFAULT_SOFT_LIMIT_SEC,
)
from jet.audio.helpers.silence import SAMPLE_RATE
from jet.audio.normalization.norm_speech_loudness import normalize_audio_for_vad
from jet.audio.normalization.quant import quantize_audio
from jet.audio.speech.segment_store import SegmentStore
from jet.audio.speech.utils import display_segments
from jet.audio.speech_detector import record_from_mic
from jet.audio.speech_handlers.base import SpeechSegmentHandler
from jet.audio.speech_handlers.global_reset_handler import GlobalResetHandler
from jet.audio.speech_handlers.short_segment_accumulator import ShortSegmentAccumulator
from jet.audio.speech_handlers.speech_events import SpeechSegmentEndEvent
from jet.audio.speech_handlers.subtitle_overlay_window import SubtitleOverlay
from jet.audio.speech_handlers.vad_firered_splitter import split_segment_with_vad
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
    verbose: bool = False,
) -> None:
    """Fire on_segment_end on every registered handler. Errors are caught per-handler."""
    seg_audio_np, _ = normalize_audio_for_vad(seg_audio_np, sample_rate)
    duration = speech_seg["duration"]
    if duration >= DEFAULT_SOFT_LIMIT_SEC:
        seg_audio_np, _ = quantize_audio(
            seg_audio_np,
            target_dtype="float16",
            sr=sample_rate,
            verbose=verbose,
        )

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


def main_live_speech_translation(
    verbose: bool = False,
    min_seg_duration_sec: float = DEFAULT_MIN_SEG_DURATION_SEC,
    max_seg_duration_sec: float = DEFAULT_MAX_SEG_DURATION_SEC,
    max_seg_gap_sec: float = DEFAULT_MAX_SEG_GAP_SEC,
    acc_max_duration_sec: float = DEFAULT_ACC_MAX_DURATION_SEC,
):
    """
    Live speech translation with segment accumulation.

    Segments are accumulated into groups based on:
    - Gaps ≤ max_seg_gap_sec (default 2.0s) are merged
    - Groups stop growing when adding the next segment would exceed acc_max_duration_sec (default 5.0s)
    - Segments ≥ max_seg_duration_sec (default 3.0s) pass through immediately as complete utterances

    Parameters
    ----------
    verbose : bool
        Enable debug logging.
    min_seg_duration_sec : float
        Minimum valid segment duration. Shorter segments are still processed but logged.
    max_seg_duration_sec : float
        Target utterance duration. Segments ≥ this pass through. Groups flush when adding
        the next segment would push the group total above this threshold.
    max_seg_gap_sec : float
        Maximum silence gap between segments to allow merging.
    acc_max_duration_sec : float
        Hard ceiling on merged group duration. Groups flush before exceeding this.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    _sigint_timer = QTimer()
    _sigint_timer.start(200)
    _sigint_timer.timeout.connect(lambda: None)

    ws_sender: WebsocketSubtitleSender = WebsocketSubtitleSender(
        global_srt_path=OUTPUT_DIR / "subtitles.srt"
    )
    handlers: list[SpeechSegmentHandler] = [ws_sender]
    global_reset_handler = GlobalResetHandler()
    all_segments_path = OUTPUT_DIR / "all_segments.json"
    subtitles_path = OUTPUT_DIR / "subtitles.srt"
    segment_store = SegmentStore(OUTPUT_DIR / "segments")
    completed_segments: list[SpeechSegment] = []
    audio_stats = {
        "total_segments": 0,
        "segments_with_overflow": 0,
        "empty_segments": 0,
        "sub_segments_created": 0,
        "segments_merged": 0,
    }

    # ── Accumulator lives at this scope so _on_clear can reset it ─────
    # Created with placeholder defaults; real instance is created in _recording_loop
    accumulator: ShortSegmentAccumulator | None = None

    def _on_clear() -> None:
        """Reset all state when the user clears the session."""
        nonlocal accumulator

        # Flush and discard any pending accumulated segments
        if accumulator is not None and accumulator.has_pending():
            logger.info(
                f"[on_clear] Discarding {accumulator.pending_duration:.3f}s of "
                f"pending accumulated segments"
            )
            accumulator.reset()

        completed_segments.clear()
        segment_store.reset()
        ws_sender.clear_queue()
        audio_stats.update(
            {
                "total_segments": 0,
                "segments_with_overflow": 0,
                "empty_segments": 0,
                "sub_segments_created": 0,
                "segments_merged": 0,
            }
        )
        logger.info("[on_clear] All state reset")

    overlay = SubtitleOverlay.create_and_connect(
        ws_sender,
        extra_clear_paths=[all_segments_path, subtitles_path],
        on_clear=_on_clear,
        global_reset_handler=global_reset_handler,
    )

    _stop_recording = threading.Event()

    def _dispatch_merged(
        merged_seg: SpeechSegment,
        merged_audio: np.ndarray,
        had_overflow: bool,
        recording_started_at: datetime,
    ) -> None:
        """Save and dispatch one (possibly merged) segment."""
        if merged_audio.size == 0:
            logger.warning("[recorder] Merged segment has empty audio — skipping")
            return

        seg_dir, seg_number = segment_store.save(
            merged_seg, merged_audio, sample_rate=SAMPLE_RATE
        )
        merged_seg["num"] = seg_number
        logger.success(
            f"Speech {seg_number}: "
            f"{merged_seg.get('start_time_utc', 'N/A')} → "
            f"{merged_seg.get('end_time_utc', 'N/A')} "
            f"dur={merged_seg.get('duration', 0.0):.3f}s "
            f"[{'⚠ OVERFLOW' if had_overflow else '✓ clean'}]"
        )
        dispatch_handlers(
            handlers,
            merged_seg,
            merged_audio,
            seg_dir,
            seg_number,
            SAMPLE_RATE,
            recording_started_at,
            verbose=verbose,
        )
        completed_segments.append(merged_seg.copy())
        save_file(completed_segments, all_segments_path)

    def _recording_loop() -> None:
        nonlocal accumulator

        recording_started_at = datetime.now(timezone.utc)
        logger.info(
            f"[recorder] Audio recording started at {recording_started_at.isoformat()}"
        )
        logger.info(
            f"[recorder] ShortSegmentAccumulator config:\n"
            f"  min_seg_duration  = {min_seg_duration_sec:.2f}s\n"
            f"  max_seg_duration  = {max_seg_duration_sec:.2f}s  (pass-through / group flush trigger)\n"
            f"  max_seg_gap       = {max_seg_gap_sec:.2f}s  (max gap to allow merging)\n"
            f"  acc_max_duration  = {acc_max_duration_sec:.2f}s  (hard ceiling on merged group)"
        )

        # Create the accumulator instance for this recording session
        accumulator = ShortSegmentAccumulator(
            min_seg_duration_sec=min_seg_duration_sec,
            max_seg_duration_sec=max_seg_duration_sec,
            max_gap_sec=max_seg_gap_sec,
            acc_max_duration_sec=acc_max_duration_sec,
            sample_rate=SAMPLE_RATE,
            verbose=verbose,
        )

        data_stream = record_from_mic(
            duration=None,
            trim_silent=False,
            quit_on_silence=False,
            verbose=verbose,
        )

        for speech_seg, seg_audio_np in data_stream:
            audio_stats["total_segments"] += 1
            had_overflow = speech_seg.get("had_overflow", False)
            if had_overflow:
                audio_stats["segments_with_overflow"] += 1
                logger.warning(
                    f"[recorder] Segment {speech_seg.get('num', '?')} affected by audio overflow! "
                    f"({audio_stats['segments_with_overflow']}/{audio_stats['total_segments']} segments affected)"
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

            # Split large segments into sub-segments using secondary VAD
            split_segments = split_segment_with_vad(
                segment=speech_seg,
                audio_np=seg_audio_np,
                sample_rate=SAMPLE_RATE,
                verbose=verbose,
            )

            if len(split_segments) > 1:
                audio_stats["sub_segments_created"] += len(split_segments) - 1
                logger.info(
                    f"[recorder] Segment split into {len(split_segments)} sub-segments"
                )

            # Feed each sub-segment through the accumulator
            for sub_seg in split_segments:
                sub_start = float(sub_seg["start"])
                sub_end = float(sub_seg["end"])
                original_start = float(speech_seg["start"])
                rel_start = sub_start - original_start
                rel_end = sub_end - original_start
                start_sample = int(round(rel_start * SAMPLE_RATE))
                end_sample = int(round(rel_end * SAMPLE_RATE))
                sub_audio = seg_audio_np[start_sample:end_sample].copy()

                if sub_audio.size == 0:
                    logger.warning(
                        f"[recorder] Sub-segment [{rel_start:.3f}s, {rel_end:.3f}s] "
                        f"has empty audio — skipping"
                    )
                    continue

                # Push to accumulator — may return merged groups ready for dispatch
                ready = accumulator.push(sub_seg, sub_audio)
                for merged_seg, merged_audio in ready:
                    # Track whether this was actually merged from multiple segments
                    is_merged = merged_seg.get("duration", 0.0) > float(
                        sub_seg.get("duration", 0.0)
                    )
                    if is_merged:
                        audio_stats["segments_merged"] += 1
                    _dispatch_merged(
                        merged_seg,
                        merged_audio,
                        had_overflow=merged_seg.get("had_overflow", False),
                        recording_started_at=recording_started_at,
                    )

        # Stream ended: flush any remaining accumulated segments
        logger.info("[recorder] Flushing remaining accumulated segments…")
        for merged_seg, merged_audio in accumulator.flush():
            audio_stats["segments_merged"] += 1
            _dispatch_merged(
                merged_seg,
                merged_audio,
                had_overflow=merged_seg.get("had_overflow", False),
                recording_started_at=recording_started_at,
            )

        quality_ok = audio_stats["segments_with_overflow"] == 0
        logger.info(
            f"[recorder] Recording complete. Stats:\n"
            f"  Total parent segments:   {audio_stats['total_segments']}\n"
            f"  Sub-segments created:    {audio_stats['sub_segments_created']}\n"
            f"  Merged (accumulated):    {audio_stats['segments_merged']}\n"
            f"  Total segments emitted:  {len(completed_segments)}\n"
            f"  With overflow:           {audio_stats['segments_with_overflow']}\n"
            f"  Empty segments:          {audio_stats['empty_segments']}\n"
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
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--min-seg",
        type=float,
        default=DEFAULT_MIN_SEG_DURATION_SEC,
        help=f"Minimum valid segment duration in seconds (default: {DEFAULT_MIN_SEG_DURATION_SEC})",
    )
    parser.add_argument(
        "--max-seg",
        type=float,
        default=DEFAULT_MAX_SEG_DURATION_SEC,
        help=f"Target utterance duration — segments ≥ this pass through, groups flush when adding next would exceed this (default: {DEFAULT_MAX_SEG_DURATION_SEC})",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=DEFAULT_MAX_SEG_GAP_SEC,
        help=f"Max gap in seconds between segments to allow merging (default: {DEFAULT_MAX_SEG_GAP_SEC})",
    )
    parser.add_argument(
        "--acc-max",
        type=float,
        default=DEFAULT_ACC_MAX_DURATION_SEC,
        help=f"Hard ceiling on merged group duration in seconds (default: {DEFAULT_ACC_MAX_DURATION_SEC})",
    )
    args = parser.parse_args()

    main_live_speech_translation(
        verbose=args.verbose,
        min_seg_duration_sec=args.min_seg,
        max_seg_duration_sec=args.max_seg,
        max_seg_gap_sec=args.max_gap,
        acc_max_duration_sec=args.acc_max,
    )
