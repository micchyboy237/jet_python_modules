"""
WebsocketSubtitleSender
========================
Implements SpeechSegmentHandler.
On each completed segment it:
  1. Builds the binary WS message  (JSON header \x00 PCM int16 audio)
  2. Sends it to the live-subtitles WebSocket server
  3. Awaits the JSON response
  4. Persists request.json, response.json, and subtitle.srt
     inside the matching segment_NNN/ directory

Architecture
------------
A dedicated asyncio event loop runs in a background daemon thread.
`on_segment_end` submits work to that loop via run_coroutine_threadsafe,
so it is safe to call from the synchronous recorder thread.

NEW: Queue status tracking & auto-retry
- Emits queue status updates to a QueueObserver (SubtitleOverlay)
- Auto-retries failed segments infinitely with exponential backoff

NEW: Queue management
- max_queue_size limits pending items (oldest dropped when full)
- max_queue_age_sec skips segments that waited too long in queue
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from jet.audio.async_utils.task_queue import AsyncTaskQueue
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.helpers.config import FRAME_LENGTH_MS, FRAME_SHIFT_MS, SAMPLE_RATE
from jet.audio.speech_handlers.api_types import (
    ClientHeader,
    ServerResponse,
    ServerSuccessResponse,
    SubtitleNotification,
)
from jet.audio.speech_handlers.base import SpeechSegmentHandler
from jet.audio.speech_handlers.language_cache import LanguageCode, LanguageStore
from jet.audio.speech_handlers.queue_observer import QueueObserver
from jet.audio.speech_handlers.speech_events import (
    SpeechSegmentEndEvent,
    SpeechSegmentStartEvent,
)
from jet.audio.speech_handlers.srt_utils import (
    build_segment_srt,
    merge_and_write_global_srt,
    write_srt,
)
from jet.audio.speech_handlers.subtitle_observer import SubtitleObserver
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import (
    ConnectionClosedError,
    ConnectionClosedOK,
)

console = Console()
logger = logging.getLogger(__name__)

_END_REASON_TO_VAD: dict[str | None, str] = {
    "silence": "silence",
    "hard_limit": "hard_limit",
    "valley": "valley",
    None: "silence",
}

# Default queue limits
DEFAULT_MAX_QUEUE_SIZE: int = 10  # Drop oldest when 10+ pending
DEFAULT_MAX_QUEUE_AGE_SEC: float = 30.0  # Skip segments older than 30s


def _vad_reason(segment: SpeechSegment) -> str:
    return _END_REASON_TO_VAD.get(segment.get("end_reason"), "silence")


def _to_pcm_int16_bytes(audio_np: np.ndarray) -> bytes:
    """
    Normalise to float32 [-1, 1], convert to int16, return raw little-endian bytes.
    Handles int16 input (passthrough) and multi-channel (mean to mono).
    """
    arr = audio_np
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    arr = arr.astype(np.float32)
    if np.issubdtype(audio_np.dtype, np.integer):
        arr = arr / np.iinfo(audio_np.dtype).max
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767).astype("<i2").tobytes()


def _build_message(header: ClientHeader, audio_bytes: bytes) -> bytes:
    """Encode: UTF-8 JSON header + null byte + raw PCM bytes."""
    return (
        json.dumps(header, ensure_ascii=False).encode("utf-8") + b"\x00" + audio_bytes
    )


def _log_request(
    seg_num: int, header: dict, audio_bytes: bytes, attempt: int = 1
) -> None:
    retry_suffix = f" (Retry #{attempt})" if attempt > 1 else ""
    title = f"[bold cyan]WS Request — Segment {seg_num}{retry_suffix}[/bold cyan]"
    table = Table(
        title=title,
        show_header=False,
    )
    table.add_column("Key", style="dim cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("seg_num", str(seg_num))
    for k, v in header.items():
        table.add_row(k, str(v))
    table.add_row("audio_bytes", f"{len(audio_bytes):,} bytes")
    if attempt > 1:
        table.add_row("retry_attempt", str(attempt))
    console.print(table)


def _log_response(response: ServerResponse, seg_num: int, attempt: int = 1) -> None:
    success = response.get("success", False)
    title_color = "bold green" if success else "bold red"
    status = "✅ success" if success else "❌ error"
    retry_suffix = f" (after {attempt} attempt(s))" if attempt > 1 else ""
    title = f"[{title_color}]WS Response — Segment {seg_num}  {status}{retry_suffix}[/{title_color}]"
    table = Table(
        title=title,
        show_header=False,
    )
    table.add_column("Key", style="dim green", no_wrap=True)
    table.add_column("Value", style="white")
    highlights = [
        "uuid",
        "success",
        "ja_text",
        "en_text",
        "context_duration",
        "new_duration",
        "coverage_label",
        "transcribed_duration_pctg",
        "new_ja_similarity",
        "error",
    ]
    for k in highlights:
        if k in response:
            table.add_row(k, str(response[k]))
    console.print(table)


def _file_link(path: Path) -> str:
    """Return a Rich markup string that is a clickable hyperlink to *path*."""
    uri = path.as_uri()
    return f"[link={uri}][cyan]{path.name}[/cyan][/link]"


def _log_saved(
    seg_dir: Path, files: list[Path], global_path: Path | None = None
) -> None:
    lines_parts: list[str] = []
    for p in files:
        lines_parts.append(f"  [dim]→[/dim] {_file_link(p)}")
    if global_path is not None:
        lines_parts.append(
            f"  [dim]→[/dim] [dim](global)[/dim] {_file_link(global_path)}"
        )
    lines = "\n".join(lines_parts)
    console.print(
        Panel(
            lines,
            title=f"[bold]Saved to [link={seg_dir.as_uri()}]{seg_dir.name}[/link][/bold]",
            expand=False,
        )
    )


def _compute_avg_vad_prob(segment: SpeechSegment) -> Optional[float]:
    """Average of per‑frame VAD probabilities (if available)."""
    probs = segment.get("segment_probs")
    if not probs:
        return None
    return float(np.mean(probs))


def _compute_speech_frames_pctg(segment: SpeechSegment) -> Optional[float]:
    """Percentage of frames with VAD probability > 0.5."""
    probs = segment.get("segment_probs")
    if not probs:
        return None
    speech_count = sum(1 for p in probs if p > 0.5)
    return (speech_count / len(probs)) * 100.0


def _compute_speech_dur_sec(
    segment: SpeechSegment, frame_dur: float = 0.01
) -> Optional[float]:
    """Total duration (seconds) of frames with VAD probability > 0.5 (default 10 ms frames)."""
    probs = segment.get("segment_probs")
    if not probs:
        return None
    speech_count = sum(1 for p in probs if p > 0.5)
    return speech_count * frame_dur


def _compute_vad_score(segment: SpeechSegment, audio_np: np.ndarray) -> Optional[float]:
    """
    Compute balanced VAD score from segment probabilities using
    score_balanced_speech from vad_segment_scorer.
    """
    from jet.audio.audio_waveform.vad.vad_segment_scorer import (
        score_balanced_speech,
    )

    probs = segment.get("segment_probs")
    if not probs:
        return None
    try:
        return score_balanced_speech(
            probs,
            audio_samples=audio_np,
        )
    except Exception as exc:
        console.print(
            f"[yellow][WS] Balanced VAD scoring with audio trimming failed: {exc}[/yellow]"
        )
        try:
            return score_balanced_speech(
                probs,
                audio_samples=None,
                sample_rate=SAMPLE_RATE,
                frame_length_ms=FRAME_LENGTH_MS,
                hop_length_ms=FRAME_SHIFT_MS,
            )
        except Exception as exc2:
            console.print(
                f"[yellow][WS] Balanced VAD scoring failed completely: {exc2}[/yellow]"
            )
            return None


class WebsocketSubtitleSender(SpeechSegmentHandler):
    """
    Thread-safe WebSocket handler.
    Keeps a single persistent connection; auto-reconnects on drops.
    Auto-retry on errors with infinite retry and exponential backoff.
    Emits queue status to a QueueObserver for UI display.

    NEW: Queue overflow protection
    - max_queue_size: drops oldest pending segment when queue is full
    - max_queue_age_sec: skips segments that waited too long in queue
    """

    def __init__(
        self,
        ws_url: str | None = None,
        reconnect_delay: float = 2.0,
        send_timeout: float = 15.0,
        global_srt_path: Path | None = None,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
        max_queue_age_sec: float = DEFAULT_MAX_QUEUE_AGE_SEC,
    ) -> None:
        _default_ws_url = (
            "ws://"
            + os.getenv("LOCAL_LIVE_SUBTITLES_HOST", "localhost:8000")
            + "/ws/live-subtitles"
        )
        self.ws_url = ws_url or _default_ws_url
        if not self.ws_url:
            raise ValueError(
                "WebSocket URL not provided. "
                "Pass ws_url= or set LOCAL_LIVE_SUBTITLES_HOST."
            )
        self.reconnect_delay = reconnect_delay
        self.send_timeout = send_timeout
        self.global_srt_path = global_srt_path
        self.max_queue_size = max_queue_size
        self.max_queue_age_sec = max_queue_age_sec
        self._language_store = LanguageStore()
        self._ws: Optional[ClientConnection] = None
        self._ws_lock = asyncio.Lock()
        self._stop_event = threading.Event()
        self._connected_event = asyncio.Event()
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop, daemon=True, name="ws-subtitle-loop"
        )
        self._loop_thread.start()
        self._loop.call_soon_threadsafe(
            lambda: self._loop.create_task(self._connection_manager())
        )
        self._task_queue = AsyncTaskQueue(
            self._loop,
            name="ws-subtitle-sender",
            maxsize=self.max_queue_size,
        )
        console.print(
            f"[bold blue][WS][/bold blue] Connecting → [cyan]{self.ws_url}[/cyan]"
        )
        self._observers: list[SubtitleObserver] = []
        self._observers_lock = threading.Lock()
        self._queue_observer: Optional[QueueObserver] = None
        self._prev_end_time_utc: Optional[str] = None
        self._max_retry_delay: float = 30.0
        self._retry_backoff_base: float = 1.5
        self._queue_status_timer: Optional[asyncio.Task] = None
        self._loop.call_soon_threadsafe(
            lambda: setattr(
                self,
                "_queue_status_timer",
                self._loop.create_task(self._periodic_queue_status()),
            )
        )
        logger.info(
            "[WebsocketSubtitleSender] Initialized with auto-retry, "
            "max_queue_size=%d, max_queue_age=%.1fs",
            self.max_queue_size,
            self.max_queue_age_sec,
        )

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _periodic_queue_status(self) -> None:
        """Periodically emit queue status to observer."""
        while not self._stop_event.is_set():
            try:
                await self._emit_queue_status()
            except Exception as exc:
                logger.error(f"Queue status emission failed: {exc}")
            await asyncio.sleep(0.5)

    async def _emit_queue_status(
        self,
        processing_seg_num: Optional[int] = None,
        status_color: str = "#8b949e",
        extra_info: Optional[dict] = None,
    ) -> None:
        """Emit current queue status to the observer with detailed info."""
        if self._queue_observer is None:
            return
        qsize = self._task_queue.qsize
        current_task = self._task_queue.current_task
        status_info = {
            "qsize": qsize,
            "maxsize": self.max_queue_size,
            "is_processing": current_task is not None,
            "segment_num": processing_seg_num,
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        }
        if extra_info:
            status_info.update(extra_info)

        if current_task is not None and processing_seg_num is not None:
            seg_dur = extra_info.get("duration", 0) if extra_info else 0
            status = f"⏳ Seg #{processing_seg_num} ({seg_dur:.1f}s) — sending..."
            color = "#58a6ff"
        elif current_task is not None and extra_info and "retry_attempt" in extra_info:
            attempt = extra_info["retry_attempt"]
            delay = extra_info.get("retry_delay", 0)
            status = (
                f"🔄 Seg #{processing_seg_num} — retry {attempt} (wait {delay:.1f}s)"
            )
            color = "#f0883e"
        elif current_task is not None:
            status = "⏳ Processing..."
            color = "#58a6ff"
        elif qsize > 0:
            status = f"📋 {qsize}/{self.max_queue_size} queued"
            color = "#e3b341" if qsize < self.max_queue_size else "#f85149"
        else:
            status = "✓ Idle"
            color = "#3fb950"
        try:
            self._queue_observer.on_queue_status(status, qsize, color, status_info)
        except Exception as exc:
            logger.error(f"Failed to update queue observer: {exc}")

    async def _connection_manager(self) -> None:
        while not self._stop_event.is_set():
            try:
                async with connect(
                    self.ws_url,
                    max_size=None,
                    compression=None,
                    ping_interval=30,
                    ping_timeout=30,
                    close_timeout=10,
                ) as ws:
                    self._ws = ws
                    self._connected_event.set()
                    console.print(
                        f"[bold green][WS][/bold green] Connected → [cyan]{self.ws_url}[/cyan]"
                    )
                    await self._emit_queue_status(status_color="#3fb950")
                    await ws.wait_closed()
            except (ConnectionClosedOK, ConnectionClosedError) as exc:
                console.print(f"[yellow][WS][/yellow] Connection closed: {exc}")
            except OSError as exc:
                console.print(f"[red][WS][/red] Network error: {exc}")
            except Exception as exc:
                console.print(
                    f"[red][WS][/red] Unexpected error: {type(exc).__name__}: {exc}"
                )
            finally:
                self._ws = None
                self._connected_event.clear()
                await self._emit_queue_status(status_color="#f85149")
            if self._stop_event.is_set():
                break
            console.print(
                f"[yellow][WS][/yellow] Reconnecting in {self.reconnect_delay:.1f}s …"
            )
            await asyncio.sleep(self.reconnect_delay)
        console.print("[dim][WS] Connection manager exited.[/dim]")

    async def _send_and_receive(
        self,
        header: ClientHeader,
        audio_bytes: bytes,
    ) -> ServerResponse:
        """
        Wait until connected, send the binary message, await the text response.
        Raises RuntimeError if no response arrives within send_timeout.
        """
        try:
            await asyncio.wait_for(
                self._connected_event.wait(), timeout=self.send_timeout
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"[WS] Timed out waiting for connection after {self.send_timeout}s"
            )
        ws = self._ws
        if ws is None:
            raise RuntimeError("[WS] Connection lost right before send")
        message = _build_message(header, audio_bytes)
        await ws.send(message)
        raw = await asyncio.wait_for(ws.recv(), timeout=self.send_timeout)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    @staticmethod
    def _save_request(seg_dir: Path, header: ClientHeader, audio_bytes: bytes) -> Path:
        payload = {**header, "audio_bytes_len": len(audio_bytes)}
        path = seg_dir / "request.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    @staticmethod
    def _save_response(seg_dir: Path, response: ServerResponse) -> Path:
        path = seg_dir / "response.json"
        path.write_text(
            json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    @staticmethod
    def _save_srt(
        seg_dir: Path,
        response: ServerSuccessResponse,
        start_sec: float,
        end_sec: float,
        seg_number: int,
    ) -> Path:
        """Write a single-block subtitle.srt: Japanese line 1, English line 2."""
        path = seg_dir / "subtitle.srt"
        srt_content = build_segment_srt(
            index=seg_number,
            start_sec=start_sec,
            end_sec=end_sec,
            ja_text=response.get("ja_text", ""),
            en_text=response.get("en_text", ""),
        )
        write_srt(path, srt_content)
        return path

    def _update_global_srt(self, seg_dir: Path) -> Path | None:
        """Re-merge all segment subtitle.srt files into the global subtitles.srt."""
        if self.global_srt_path is None:
            return None
        segments_root = seg_dir.parent
        merge_and_write_global_srt(segments_root, self.global_srt_path)
        return self.global_srt_path

    def on_segment_start(self, event: SpeechSegmentStartEvent) -> None:
        """Nothing to do on start — reserved for future use."""
        pass

    def on_segment_end(self, event: SpeechSegmentEndEvent) -> None:
        """
        Called from the recorder thread.
        Enqueues _handle_segment for serial execution via AsyncTaskQueue.
        Returns immediately — recorder thread is never blocked.
        """
        self._task_queue.enqueue(self._handle_segment(event))

    def add_observer(self, observer: SubtitleObserver) -> None:
        """Register an observer to receive every ServerResponse."""
        with self._observers_lock:
            if observer not in self._observers:
                self._observers.append(observer)

    def remove_observer(self, observer: SubtitleObserver) -> None:
        """Unregister a previously added observer."""
        with self._observers_lock:
            self._observers = [o for o in self._observers if o is not observer]

    def set_queue_observer(self, observer: QueueObserver) -> None:
        """
        Set a QueueObserver to receive queue and retry status updates.
        Usually the SubtitleOverlay.
        """
        self._queue_observer = observer
        logger.info(
            f"[WebsocketSubtitleSender] Queue observer set: {type(observer).__name__}"
        )

    def _notify_observers(self, response: ServerResponse) -> None:
        """Call on_subtitle_response on all registered observers (errors caught per-observer)."""
        with self._observers_lock:
            observers = list(self._observers)
        for observer in observers:
            try:
                observer.on_subtitle_response(response)
            except Exception as exc:
                console.print(
                    f"[yellow][WS][/yellow] Observer {type(observer).__name__} failed: {exc}"
                )

    async def _handle_segment_with_retry(self, event: SpeechSegmentEndEvent) -> None:
        """
        Process a segment with infinite auto-retry on failure.
        Skips retry for permanent failures (no_speech, empty text, specific error codes).
        Uses exponential backoff with a max delay cap.

        NEW: Skips stale segments that waited too long in the queue.
        """
        # --- Staleness check: skip segments that waited too long in queue ---
        queue_wait = time.monotonic() - event.enqueued_at
        if queue_wait > self.max_queue_age_sec:
            logger.warning(
                "[WebsocketSubtitleSender] Segment %d skipped — waited %.1fs in queue "
                "(max_queue_age=%.1fs). Queue depth: %d",
                event.segment_number,
                queue_wait,
                self.max_queue_age_sec,
                self._task_queue.qsize,
            )
            console.print(
                f"[yellow][WS][/yellow] Segment {event.segment_number} "
                f"skipped — stale ({queue_wait:.1f}s in queue > {self.max_queue_age_sec:.1f}s max)"
            )
            if self._queue_observer:
                try:
                    self._queue_observer.on_queue_status(
                        f"⏭️ Seg #{event.segment_number} skipped (stale {queue_wait:.1f}s)",
                        self._task_queue.qsize,
                        "#8b949e",
                        {
                            "segment_num": event.segment_number,
                            "queue_wait_sec": round(queue_wait, 2),
                            "max_queue_age_sec": self.max_queue_age_sec,
                            "status": "skipped_stale",
                        },
                    )
                except Exception:
                    pass
            return

        seg: SpeechSegment = event.segment
        seg_num: int = event.segment_number
        if event.audio_np is None or event.audio_np.size == 0:
            console.print(f"[dim][WS][/dim] Segment {seg_num} skipped — empty audio")
            return

        audio_bytes = _to_pcm_int16_bytes(event.audio_np)
        if len(audio_bytes) == 0:
            console.print(f"[dim][WS][/dim] Segment {seg_num} skipped — zero PCM bytes")
            return

        seg_dir: Path = event.segment_dir
        start_sec = float(seg["start"])
        end_sec = float(seg["end"])
        duration = float(seg["duration"])
        start_time_utc: Optional[str] = seg.get("start_time_utc")
        end_time_utc: Optional[str] = seg.get("end_time_utc")
        gap_sec: Optional[float] = None
        if self._prev_end_time_utc is not None and start_time_utc is not None:
            from datetime import datetime

            try:
                prev_end_dt = datetime.fromisoformat(self._prev_end_time_utc)
                start_dt = datetime.fromisoformat(start_time_utc)
                gap_sec = round((start_dt - prev_end_dt).total_seconds(), 4)
            except (ValueError, TypeError):
                pass
        self._prev_end_time_utc = end_time_utc

        vad_score = _compute_vad_score(seg, event.audio_np)
        segment_id = f"segment_{seg_num:03d}"
        header: ClientHeader = {
            "uuid": str(uuid.uuid4()),
            "sample_rate": event.sample_rate,
            "duration_sec": round(duration, 4),
            "start_sec": round(start_sec, 4),
            "end_sec": round(end_sec, 4),
            "vad_reason": _vad_reason(seg),
            "forced": _vad_reason(seg) != "silence",
            "started_at": event.started_at.isoformat(),
            "start_time_utc": start_time_utc,
            "end_time_utc": end_time_utc,
            "gap_sec": gap_sec,
            "vad_score": vad_score,
            "language": self._language_store.language,
            "segment_number": seg_num,
            "segment_id": segment_id,
        }

        attempt = 0
        last_error = None
        response = None
        final_success = False
        permanent_failure_reason: Optional[str] = None

        while True:
            attempt += 1
            if attempt == 1:
                await self._emit_queue_status(
                    processing_seg_num=seg_num,
                    status_color="#58a6ff",
                    extra_info={
                        "duration": duration,
                        "start_sec": start_sec,
                        "queue_wait_sec": round(queue_wait, 2),
                    },
                )
            else:
                delay = min(
                    self._retry_backoff_base ** (attempt - 1), self._max_retry_delay
                )
                if self._queue_observer:
                    try:
                        self._queue_observer.on_retry_status(
                            seg_num,
                            attempt,
                            delay,
                            extra_info={
                                "last_error": str(last_error)
                                if last_error
                                else "Unknown",
                                "duration": duration,
                            },
                        )
                    except Exception:
                        pass
                await self._emit_queue_status(
                    processing_seg_num=seg_num,
                    status_color="#f0883e",
                    extra_info={
                        "retry_attempt": attempt,
                        "retry_delay": delay,
                        "duration": duration,
                    },
                )
                console.print(
                    f"[yellow][WS][/yellow] Retrying segment {seg_num} "
                    f"(attempt {attempt}) after {delay:.1f}s delay..."
                )
                await asyncio.sleep(delay)

            _log_request(seg_num, header, audio_bytes, attempt)
            try:
                response = await self._send_and_receive(header, audio_bytes)
                _log_response(response, seg_num, attempt)
                if response.get("success") and "ja_text" in response:
                    console.print(
                        f"[bold green][WS][/bold green] Segment {seg_num} succeeded "
                        f"{'after ' + str(attempt) + ' attempts' if attempt > 1 else 'on first attempt'}"
                    )
                    final_success = True
                    break

                permanent_failure_reason = self._is_permanent_failure(response)
                if permanent_failure_reason:
                    console.print(
                        f"[yellow][WS][/yellow] Segment {seg_num} — {permanent_failure_reason} "
                        f"(attempt {attempt}). Saving response and skipping retry."
                    )
                    final_success = False
                    break

                error_msg = response.get("error", "Unknown server error")
                console.print(
                    f"[bold red][WS][/bold red] Segment {seg_num} failed: {error_msg} "
                    f"(attempt {attempt}, will retry)"
                )
                last_error = RuntimeError(f"Server error: {error_msg}")
            except (ConnectionClosedError, ConnectionClosedOK) as exc:
                console.print(
                    f"[yellow][WS][/yellow] Connection lost during segment {seg_num}: {exc}"
                )
                last_error = exc
            except OSError as exc:
                console.print(
                    f"[red][WS][/red] Network error for segment {seg_num}: {exc}"
                )
                last_error = exc
            except Exception as exc:
                console.print(
                    f"[bold red][WS][/bold red] Segment {seg_num} send/receive failed: "
                    f"{type(exc).__name__}: {exc} (attempt {attempt}, will retry)"
                )
                last_error = exc

        if final_success:
            await self._emit_queue_status(
                processing_seg_num=seg_num,
                status_color="#3fb950",
                extra_info={
                    "duration": duration,
                    "start_sec": start_sec,
                    "status": "success",
                    "attempts": attempt,
                },
            )
        else:
            error_msg = (
                permanent_failure_reason or str(last_error)
                if last_error
                else "Unknown error"
            )
            await self._emit_queue_status(
                processing_seg_num=seg_num,
                status_color="#f85149",
                extra_info={
                    "duration": duration,
                    "start_sec": start_sec,
                    "status": "error",
                    "attempts": attempt,
                    "error": error_msg,
                },
            )

        notification: SubtitleNotification = {
            "segment": seg,
            "num": seg_num,
            **response,
            "start_sec": round(start_sec, 4),
            "end_sec": round(end_sec, 4),
            "end_reason": _vad_reason(seg),
            "segment_dir": str(seg_dir),
            "avg_vad_prob": _compute_avg_vad_prob(seg),
            "vad_score": vad_score,
            "speech_frames_pctg": _compute_speech_frames_pctg(seg),
            "speech_dur_sec": _compute_speech_dur_sec(seg),
            "start_time_utc": start_time_utc,
            "end_time_utc": end_time_utc,
        }
        self._notify_observers(notification)

        req_path = self._save_request(seg_dir, header, audio_bytes)
        resp_path = self._save_response(seg_dir, response)
        saved_files = [req_path, resp_path]
        if response.get("success") and "ja_text" in response:
            srt_path = self._save_srt(seg_dir, response, start_sec, end_sec, seg_num)
            saved_files.append(srt_path)
        global_path = self._update_global_srt(seg_dir)
        _log_saved(seg_dir, saved_files, global_path)

    def _is_permanent_failure(self, response: dict) -> Optional[str]:
        """
        Determine if a server response indicates a permanent failure
        that won't be resolved by retrying the same audio.
        Returns:
            A string describing why it's permanent, or None if retryable.
        """
        if response.get("coverage_label") == "no_speech":
            return "Server reported no_speech"
        ja_text = response.get("ja_text", "").strip()
        en_text = response.get("en_text", "").strip()
        if not response.get("success") and not ja_text and not en_text:
            return (
                "Server returned empty text (audio may not contain translatable speech)"
            )
        error_code = response.get("error_code", "")
        if error_code in (
            "UNSUPPORTED_AUDIO",
            "NO_SPEECH_DETECTED",
            "LANGUAGE_NOT_SUPPORTED",
            "MODEL_ERROR",
            "INVALID_AUDIO_FORMAT",
        ):
            return f"Server error code: {error_code}"
        coverage = response.get("transcribed_duration_pctg")
        if isinstance(coverage, (int, float)) and coverage < 5.0:
            return f"Coverage too low ({coverage:.1f}%)"
        return None

    async def _handle_segment(self, event: SpeechSegmentEndEvent) -> None:
        """
        Wrapper that delegates to _handle_segment_with_retry.
        This is the entry point called by the task queue.
        """
        await self._handle_segment_with_retry(event)

    def set_language(self, language: "LanguageCode") -> None:
        """
        Update the transcription language.
        Safe to call from any thread — the next segment will use the new value.
        """
        self._language_store.set_language(language)

    def clear_queue(self) -> None:
        """
        Drop all pending (not yet started) subtitle tasks.
        Safe to call from any thread — e.g. from the UI clear action.
        The currently running send/receive is allowed to complete normally.
        Also resets gap tracking so the next segment starts fresh.
        """
        self._task_queue.clear()
        self._prev_end_time_utc = None
        asyncio.run_coroutine_threadsafe(
            self._emit_queue_status(status_color="#3fb950"), self._loop
        )

    def close(self) -> None:
        self._stop_event.set()
        self._task_queue.cancel()
        try:
            self._task_queue.drain_sync(timeout=self.send_timeout + 5)
        except Exception:
            pass
        if self._queue_status_timer is not None:
            self._queue_status_timer.cancel()
        if self._ws is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
                future.result(timeout=5.0)
            except Exception as exc:
                console.print(f"[yellow][WS][/yellow] Error during close: {exc}")
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5.0)
        console.print("[dim][WS] Shutdown complete.[/dim]")
