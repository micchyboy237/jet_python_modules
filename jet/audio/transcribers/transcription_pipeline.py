# transcription_pipeline.py
from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional, Callable, Dict

from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

import logging
import asyncio
import random
import string

from jet.audio.transcribers.base import AudioInput, load_audio
from jet.audio.transcribers.base_client_async import atranscribe_audio  # async endpoint

logging.basicConfig(level="DEBUG", handlers=[RichHandler()], force=True)
logger = logging.getLogger("TranscriptionPipeline")
logger.setLevel(logging.DEBUG)

console = Console()

def _random_suffix(length=4):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

async def transcribe_ja_chunk(audio: AudioInput) -> dict:
    """
    Call the unified transcription+translation endpoint asynchronously.
    Returns a dict with transcription, translation, and word-level timestamps.
    Adds a random 4 char suffix (with underscore) to filename.
    """
    suffix = _random_suffix()
    filename = f"segment_live_{suffix}.wav"
    result = await atranscribe_audio(audio, filename=filename)
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
        on_result: Optional[Callable[[str, str, list[dict], Dict[str, Any]], None]] = None,
    ):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="TransPipe")
        self._queue: deque[Future[None]] = deque()
        self._lock = threading.Lock()
        self._cache: dict[AudioKey, tuple[str, str, list[dict]]] = {}
        self._cache_order: deque[AudioKey] = deque(maxlen=cache_size)
        self.on_result = on_result
        logger.debug("Pipeline init: max_workers=%d cache_size=%d", max_workers, cache_size)

        self._loop_thread = None
        self._loop = None
        self._start_loop()

    def _start_loop(self):
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        import threading
        self._loop_thread = threading.Thread(
            target=run_loop,
            name="PipelineAsyncLoop",
            daemon=True,
        )
        self._loop_thread.start()

        # Give it a moment to start
        import time
        time.sleep(0.05)

    def _make_key(self, audio: AudioInput) -> AudioKey:
        audio = load_audio(audio)
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

    def submit_segment(self, audio: AudioInput, **custom_args: Any) -> None:
        key = self._make_key(audio)
        logger.debug("submit_segment → key=%s (duration=%.3fs) custom_args=%s", key, key.duration_sec, custom_args)

        if (cached := self._cache_get(key)) is not None:
            ja_text, en_text, timestamps = cached
            logger.debug("CACHE HIT → immediate callback with custom_args")
            self._print_result(ja_text, en_text)
            if self.on_result:
                self.on_result(ja_text, en_text, timestamps, custom_args.copy())
            return

        logger.debug("CACHE MISS → submitting to event loop with custom_args=%s", custom_args)

        coro = self._process(audio, key, custom_args)
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        # Optional: wrap in concurrent.futures Future if you need .result() later

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

    async def _process(self, audio: AudioInput, key: AudioKey, custom_args: Dict[str, Any]) -> None:
        logger.debug("START _process → key=%s custom_args=%s", key, custom_args)
        try:
            result = await transcribe_ja_chunk(audio)
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
                self.on_result(ja_text, en_text, timestamps, custom_args.copy())

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
