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
  await queue.drain()                   # optional — wait until empty
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
        self._queue: asyncio.Queue[Coroutine[Any, Any, Any] | None] = asyncio.Queue(
            maxsize=maxsize
        )
        # Worker task is created inside the loop thread so the Queue is
        # bound to the correct loop.
        self._worker_future = asyncio.run_coroutine_threadsafe(
            self._start_worker(), loop
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, coro: Coroutine[Any, Any, Any]) -> None:
        """
        Submit *coro* for serial execution.

        Thread-safe — may be called from any thread.
        Returns immediately; the coroutine runs in the background.
        """
        self._loop.call_soon_threadsafe(lambda: self._loop.create_task(self._put(coro)))

    def cancel(self) -> None:
        """
        Signal the worker to stop after finishing the current task.

        Thread-safe.  Outstanding queued items are discarded.
        """
        # Sending None is the sentinel that tells the worker to exit.
        self._loop.call_soon_threadsafe(lambda: self._loop.create_task(self._put(None)))

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

    async def _put(self, item: Coroutine[Any, Any, Any] | None) -> None:
        await self._queue.put(item)

    async def _start_worker(self) -> None:
        """Create the queue on the correct loop, then run the worker."""
        # Re-create the queue here so it's bound to *this* loop.
        self._queue = asyncio.Queue(maxsize=self._queue.maxsize)
        await self._worker()

    async def _worker(self) -> None:
        logger.debug("[%s] Worker started.", self._name)
        while True:
            coro = await self._queue.get()
            if coro is None:  # sentinel → shut down
                self._queue.task_done()
                logger.debug("[%s] Worker received stop sentinel.", self._name)
                break
            try:
                await coro
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[%s] Task raised %s: %s",
                    self._name,
                    type(exc).__name__,
                    exc,
                )
            finally:
                self._queue.task_done()
        logger.debug("[%s] Worker stopped.", self._name)
