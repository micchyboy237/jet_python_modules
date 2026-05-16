"""
AsyncTaskQueue
==============
A reusable, loop-bound FIFO queue that runs async callables one at a time.

Why:
  websockets raises ConcurrencyError when two coroutines call send() or
  recv() simultaneously on the same connection.  By funnelling every
  coroutine through this queue we guarantee serial execution without
  blocking the caller.

Usage:
  queue = AsyncTaskQueue(loop)          # once, at startup
  queue.enqueue(my_coroutine())         # thread-safe, non-blocking
  queue.drain_sync()                    # optional — block until empty
  queue.clear()                         # drop pending items, keep worker alive
  queue.cancel()                        # stop the worker
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class AsyncTaskQueue:
    """
    Serialises async callables on a given event loop.

    Parameters
    ----------
    loop:
        The asyncio event loop the queue worker will run on.
        Must already be running in a background thread.
    maxsize:
        Maximum number of pending items (0 = unlimited).
    name:
        Human-readable label used in log messages.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        maxsize: int = 0,
        name: str = "AsyncTaskQueue",
    ) -> None:
        self._loop = loop
        self._name = name
        self._maxsize = maxsize
        self._current_coro: Coroutine[Any, Any, Any] | None = None
        self._current_coro_name: str | None = None
        self._queue: asyncio.Queue[Coroutine[Any, Any, Any] | None] = (
            asyncio.run_coroutine_threadsafe(self._create_queue(), loop).result(
                timeout=5
            )
        )
        self._worker_future = asyncio.run_coroutine_threadsafe(self._worker(), loop)

    @property
    def current_task(self) -> Coroutine[Any, Any, Any] | None:
        """
        The coroutine currently being awaited by the worker, or None.
        Read-only. May be inspected from any thread — the reference is
        set/cleared only by the worker on the event loop thread, so reads
        from other threads are safe for inspection but should not be awaited.
        """
        return self._current_coro

    # ------------------------------------------------------------------
    # Public API (all thread-safe)
    # ------------------------------------------------------------------

    def enqueue(self, coro: Coroutine[Any, Any, Any]) -> None:
        """
        Submit *coro* for serial execution.
        Thread-safe — may be called from any thread.
        Returns immediately; the coroutine runs in the background.
        """
        self._loop.call_soon_threadsafe(lambda: self._loop.create_task(self._put(coro)))

    def clear(self) -> None:
        """
        Discard all pending (not yet started) items from the queue.
        The currently running task is allowed to finish normally.
        Thread-safe — blocks the calling thread until the drain is done
        (typically completes in microseconds).
        """
        asyncio.run_coroutine_threadsafe(self._drain_queue(), self._loop).result(
            timeout=5
        )

    def cancel(self) -> None:
        """
        Signal the worker to stop after finishing the current task.
        Thread-safe.  Outstanding queued items are discarded.
        """
        self._loop.call_soon_threadsafe(lambda: self._loop.create_task(self._put(None)))

    def drain_sync(self, timeout: float = 30.0) -> None:
        """
        Block the calling thread until the queue is empty and the
        current task is done.
        Thread-safe — may be called from any thread.
        """
        asyncio.run_coroutine_threadsafe(self._queue.join(), self._loop).result(
            timeout=timeout
        )

    async def drain(self) -> None:
        """
        Await until the queue is empty and the current task is done.
        Must be awaited from *within* the queue's event loop.
        """
        await self._queue.join()

    @property
    def qsize(self) -> int:
        """Approximate number of pending items (not including the running task)."""
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _create_queue(self) -> asyncio.Queue[Coroutine[Any, Any, Any] | None]:
        """Instantiate the queue on the event loop so it is loop-bound correctly."""
        return asyncio.Queue(maxsize=self._maxsize)

    async def _put(self, item: Coroutine[Any, Any, Any] | None) -> None:
        await self._queue.put(item)

    async def _drain_queue(self) -> None:
        if self._current_coro_name:
            logger.debug(
                "[%s] clear() called while running: %s (will finish normally)",
                self._name,
                self._current_coro_name,
            )
        dropped = 0
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item is not None:
                    item.close()
                self._queue.task_done()
                dropped += 1
            except asyncio.QueueEmpty:
                break
        if dropped:
            logger.debug(
                "[%s] clear() dropped %d pending item(s).", self._name, dropped
            )

    async def _worker(self) -> None:
        logger.debug("[%s] Worker started.", self._name)
        while True:
            coro = await self._queue.get()
            if coro is None:
                self._queue.task_done()
                logger.debug("[%s] Worker received stop sentinel.", self._name)
                break
            self._current_coro = coro
            self._current_coro_name = getattr(coro, "__qualname__", repr(coro))
            logger.debug("[%s] Starting task: %s", self._name, self._current_coro_name)
            try:
                await coro
            except Exception as exc:
                logger.error(
                    "[%s] Task raised %s: %s",
                    self._name,
                    type(exc).__name__,
                    exc,
                )
            finally:
                self._current_coro = None
                self._current_coro_name = None
                self._queue.task_done()
        logger.debug("[%s] Worker stopped.", self._name)
