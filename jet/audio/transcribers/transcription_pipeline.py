# transcription_pipeline.py
from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional, Callable

import numpy as np
from numpy.typing import NDArray
from rich.console import Console
from rich.table import Table

import logging

from jet.audio.transcribers.base_client import transcribe_audio  # new unified endpoint

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("TranscriptionPipeline")
logger.setLevel(logging.DEBUG)

console = Console()

# ----------------------------------------------------------------------
# Placeholder stubs – replace with your real implementations
# ----------------------------------------------------------------------
def transcribe_ja_chunk(audio: NDArray[np.float32]) -> dict:
    """
    Call the unified transcription+translation endpoint.
    Returns the full TranscribeResponse dict (contains both transcription and translation and word-level timestamps).
    """
    result = transcribe_audio(audio.tobytes())
    return {
        "transcription": result["transcription"],
        "translation": result["translation"],
        "words": result.get("words", []),
    }

# ----------------------------------------------------------------------
# Non-blocking pipeline with caching & deduplication
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AudioKey:
    """Immutable hashable key for audio caching."""
    hash: int
    duration_sec: float

class TranscriptionPipeline:
    def __init__(
        self,
        max_workers: int = 2,
        cache_size: int = 500,
        on_result: Optional[Callable[[str, str, list[dict]], None]] = None,
    ):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="TransPipe")
        self._queue: deque[Future[None]] = deque()
        self._lock = threading.Lock()
        self._cache: dict[AudioKey, tuple[str, str, list[dict]]] = {}
        self._cache_order: deque[AudioKey] = deque(maxlen=cache_size)
        self.on_result = on_result
        logger.debug("Pipeline init: max_workers=%d cache_size=%d", max_workers, cache_size)

    def _make_key(self, audio: NDArray[np.float32]) -> AudioKey:
        duration = len(audio) / 16000.0
        h = hash(audio.tobytes())
        return AudioKey(hash=h, duration_sec=round(duration, 3))

    def _cache_get(self, key: AudioKey) -> Optional[tuple[str, str, list[dict]]]:
        with self._lock:
            return self._cache.get(key)

    def _cache_set(self, key: AudioKey, ja: str, en: str, timestamps: list[dict]) -> None:
        with self._lock:
            logger.debug("CACHE SET attempt → key=%s ja='%s' en='%s' (words: %d)", key, ja, en, len(timestamps))
            if key in self._cache:
                logger.debug("  → key already exists, removing from order")
                self._cache_order.remove(key)
            self._cache[key] = (ja, en, timestamps)
            self._cache_order.append(key)
            logger.debug("  → cache size after set: %d / %d", len(self._cache), self._cache_order.maxlen or 0)

            while len(self._cache) > (self._cache_order.maxlen or 0):
                old_key = self._cache_order.popleft()
                logger.debug("  → EVICTING LRU key: %s", old_key)
                self._cache.pop(old_key, None)
            logger.debug("  → final cache keys: %s", list(self._cache.keys()))

    def submit_segment(self, audio: NDArray[np.float32]) -> None:
        key = self._make_key(audio)
        logger.debug("submit_segment → key=%s (duration=%.3fs)", key, key.duration_sec)

        if (cached := self._cache_get(key)) is not None:
            ja_text, en_text, timestamps = cached
            logger.debug("CACHE HIT → immediate print and optional callback")
            self._print_result(ja_text, en_text)
            if self.on_result:
                self.on_result(ja_text, en_text, timestamps)
            return

        logger.debug("CACHE MISS → submitting to executor")

        future = self._executor.submit(self._process, audio, key)

        def _cleanup(fut: Future) -> None:
            with self._lock:
                if fut in self._queue:
                    self._queue.remove(fut)
                    logger.debug("Future removed from queue → remaining: %d", len(self._queue))
                else:
                    logger.debug("Future already removed (or never added)")

        future.add_done_callback(_cleanup)

        with self._lock:
            self._queue.append(future)
            logger.debug("Future added → queue size: %d", len(self._queue))

    def _process(self, audio: NDArray[np.float32], key: AudioKey) -> None:
        logger.debug("START _process → key=%s", key)
        try:
            result = transcribe_ja_chunk(audio)
            logger.debug("transcribe_audio returned: %r", result)

            ja_text = result["transcription"].strip()
            en_text = result.get("translation", "").strip()
            timestamps = result.get("words", result.get("segments", []))

            logger.debug("Extracted ja_text: %r", ja_text)
            logger.debug("Extracted en_text: %r", en_text)
            logger.debug("Extracted word timestamps: %r", timestamps)

            logger.debug("Calling _cache_set...")
            self._cache_set(key, ja_text, en_text, timestamps)

            logger.debug("Calling _print_result...")
            self._print_result(ja_text, en_text)
            if self.on_result:
                self.on_result(ja_text, en_text, timestamps)

        except Exception as exc:
            logger.exception("Exception in _process → will re-raise")
            console.print(f"[red]Transcription pipeline error:[/] {exc}")
            raise
        finally:
            logger.debug("END _process → future will complete")
            return

    def _print_result(self, ja: Any, en: Any) -> None:
        ja_str = str(ja).strip() if not isinstance(ja, str) else ja.strip()
        en_str = str(en).strip() if not isinstance(en, str) else en.strip()

        table = Table(show_header=True, header_style="bold magenta", border_style="bright_black")
        table.add_column("Japanese", style="cyan")
        table.add_column("English", style="green")
        table.add_row(ja_str or "[dim](empty)[/]", en_str or "[dim](empty)[/]")
        console.print("\n")
        console.rule("[bold blue]Live Translation[/]")
        console.print(table)
        console.print("\n")

    def shutdown(self, wait: bool = True) -> None:
        if not wait:
            with self._lock:
                for future in list(self._queue):
                    future.cancel()
                self._queue.clear()

        self._executor.shutdown(wait=wait)

        # Final safety net
        if not wait:
            with self._lock:
                self._queue.clear()
