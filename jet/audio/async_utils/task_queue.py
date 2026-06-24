"""
AsyncTaskQueue
==============
A reusable, loop-bound FIFO queue that runs async callables one at a time.
Why:
  websockets raises ConcurrencyError when two coroutines call send() or
  recv() simultaneously on the same connection.  By funnelling every
  coroutine through this queue we guarantee serial execution without
  blocking the caller.

Key improvements over the original:
  - enqueue() is non-blocking (fire-and-forget)
  - enqueue_blocking() available when synchronous insertion is required
  - clear() uses a generation counter so in-flight tasks can detect cancellation
  - When maxsize > 0 and queue is full, the oldest pending item is dropped
    before inserting the new one (drop-oldest eviction)
  - No 5-second timeout blocks anywhere in the hot path
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
        When > 0 and the queue is full, enqueue() drops the oldest
        pending item before inserting the new one.
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

        # The asyncio.Queue must be created on the event loop thread.
        # We use maxsize=0 (unlimited) internally because we handle
        # the size limit ourselves in _put_nonblocking with drop-oldest logic.
        self._queue: asyncio.Queue[tuple[int, Coroutine[Any, Any, Any] | None]] = (
            asyncio.run_coroutine_threadsafe(self._create_queue(), loop).result(
                timeout=10
            )
        )

        # Generation counter: incremented on each clear().
        # Tasks enqueued before the clear will find their generation
        # doesn't match and can abort early.
        self._generation: int = 0

        # Track queue size manually since we use unlimited internal queue.
        self._qsize: int = 0
        self._qsize_lock = asyncio.Lock()

        self._worker_future = asyncio.run_coroutine_threadsafe(self._worker(), loop)
        logger.debug(
            "[%s] Initialized (maxsize=%d, drop_oldest=True)",
            self._name,
            self._maxsize,
        )

    # ---- Public API --------------------------------------------------------

    @property
    def current_task(self) -> Coroutine[Any, Any, Any] | None:
        """
        The coroutine currently being awaited by the worker, or None.
        Read-only. Safe to inspect from any thread.
        """
        return self._current_coro

    @property
    def generation(self) -> int:
        """Current generation counter (incremented on each clear)."""
        return self._generation

    @property
    def qsize(self) -> int:
        """Approximate number of pending items (not including the running task)."""
        return self._qsize

    def enqueue(self, coro: Coroutine[Any, Any, Any]) -> None:
        """
        Submit *coro* for serial execution.

        **Non-blocking** — schedules the put on the event loop and returns
        immediately.  If maxsize > 0 and the queue is full, the **oldest**
        pending item is dropped before inserting this one.

        NOTE: Because this is fire-and-forget, a clear() call that happens
        *immediately* after enqueue() may not see the item if the event loop
        hasn't processed the put yet.  Use enqueue_blocking() if you need
        that guarantee.
        """
        coro_name = getattr(coro, "__qualname__", repr(coro))
        asyncio.run_coroutine_threadsafe(self._put_nonblocking(coro), self._loop)
        logger.debug(
            "[%s] enqueue() scheduled (non-blocking): %s",
            self._name,
            coro_name,
        )

    def enqueue_blocking(
        self, coro: Coroutine[Any, Any, Any], timeout: float = 5.0
    ) -> None:
        """
        Submit *coro* for serial execution.

        **Blocks** the calling thread until the coroutine is physically
        inside the queue.  Use this only when you need a hard guarantee
        that a subsequent clear() will see and drop this item.
        """
        coro_name = getattr(coro, "__qualname__", repr(coro))
        asyncio.run_coroutine_threadsafe(self._put_blocking(coro), self._loop).result(
            timeout=timeout
        )
        logger.debug(
            "[%s] enqueue_blocking() completed: %s",
            self._name,
            coro_name,
        )

    def clear(self) -> None:
        """
        Discard all pending items and increment the generation counter.

        - Currently running task is allowed to finish normally, but it can
          inspect `self._generation` to decide whether to abort early.
        - All pending items are drained and their .close() method is called
          if present.
        - Blocks the calling thread until the drain completes.
        """
        asyncio.run_coroutine_threadsafe(self._drain_and_bump(), self._loop).result(
            timeout=5
        )
        logger.debug(
            "[%s] clear() completed — generation %d, qsize=%d",
            self._name,
            self._generation,
            self._qsize,
        )

    def cancel(self) -> None:
        """Signal the worker to stop. Thread-safe."""
        asyncio.run_coroutine_threadsafe(self._put_sentinel(), self._loop).result(
            timeout=5
        )
        logger.debug("[%s] cancel() sent stop sentinel", self._name)

    def drain_sync(self, timeout: float = 30.0) -> None:
        """
        Block the calling thread until the queue is empty and the
        current task is done.
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

    # ---- Internal (event-loop side) ---------------------------------------

    async def _create_queue(
        self,
    ) -> asyncio.Queue[tuple[int, Coroutine[Any, Any, Any] | None]]:
        """Instantiate the queue on the event loop so it is loop-bound correctly."""
        # Always use unlimited queue internally — we manage the size cap ourselves.
        return asyncio.Queue(maxsize=0)

    async def _put_blocking(self, item: Coroutine[Any, Any, Any] | None) -> None:
        """Put an item onto the queue and track size. Blocks if needed (sentinel only)."""
        await self._queue.put((self._generation, item))
        async with self._qsize_lock:
            self._qsize += 1
        logger.debug("[%s] _put_blocking: qsize=%d", self._name, self._qsize)

    async def _put_nonblocking(self, coro: Coroutine[Any, Any, Any]) -> None:
        """
        Put a coroutine onto the queue. If maxsize > 0 and the queue is full,
        drop the oldest pending item first (drop-oldest eviction).
        """
        coro_name = getattr(coro, "__qualname__", repr(coro))

        # Check if we need to evict
        if self._maxsize > 0 and self._qsize >= self._maxsize:
            dropped = await self._drop_oldest()
            if dropped:
                logger.warning(
                    "[%s] Queue full (maxsize=%d) — dropped oldest pending item "
                    "to make room for: %s",
                    self._name,
                    self._maxsize,
                    coro_name,
                )

        await self._queue.put((self._generation, coro))
        async with self._qsize_lock:
            self._qsize += 1
        logger.debug(
            "[%s] _put_nonblocking: enqueued %s, qsize=%d/%s",
            self._name,
            coro_name,
            self._qsize,
            self._maxsize if self._maxsize > 0 else "∞",
        )

    async def _drop_oldest(self) -> bool:
        """
        Remove and close the oldest pending item from the queue.
        Returns True if an item was dropped, False if the queue was empty.
        """
        try:
            gen, old_coro = self._queue.get_nowait()
            if old_coro is not None:
                old_name = getattr(old_coro, "__qualname__", repr(old_coro))
                try:
                    old_coro.close()
                    logger.debug(
                        "[%s] Closed dropped coroutine: %s",
                        self._name,
                        old_name,
                    )
                except Exception as exc:
                    logger.debug(
                        "[%s] Dropped coroutine %s raised on close(): %s",
                        self._name,
                        old_name,
                        exc,
                    )
            self._queue.task_done()
            async with self._qsize_lock:
                self._qsize -= 1
            logger.debug(
                "[%s] _drop_oldest: removed gen=%d, qsize=%d",
                self._name,
                gen,
                self._qsize,
            )
            return True
        except asyncio.QueueEmpty:
            logger.debug(
                "[%s] _drop_oldest: queue was empty, nothing dropped", self._name
            )
            return False

    async def _put_sentinel(self) -> None:
        """Put the stop sentinel onto the queue (always allowed, even if full)."""
        await self._queue.put((self._generation, None))
        async with self._qsize_lock:
            self._qsize += 1
        logger.debug("[%s] _put_sentinel: qsize=%d", self._name, self._qsize)

    async def _drain_and_bump(self) -> None:
        """Drain all pending items and bump the generation counter."""
        if self._current_coro_name:
            logger.debug(
                "[%s] clear() called while running: %s (will finish normally)",
                self._name,
                self._current_coro_name,
            )

        # Bump the generation so future tasks know they're from a new era.
        self._generation += 1
        dropped = 0

        while not self._queue.empty():
            try:
                gen, item = self._queue.get_nowait()
                if item is not None:
                    try:
                        item.close()
                    except Exception:
                        pass
                self._queue.task_done()
                dropped += 1
                async with self._qsize_lock:
                    self._qsize -= 1
            except asyncio.QueueEmpty:
                break

        if dropped:
            logger.debug(
                "[%s] clear() dropped %d pending item(s) — generation now %d, qsize=%d",
                self._name,
                dropped,
                self._generation,
                self._qsize,
            )
        else:
            logger.debug(
                "[%s] clear() — nothing to drop, generation now %d",
                self._name,
                self._generation,
            )

    async def _worker(self) -> None:
        logger.debug("[%s] Worker started.", self._name)
        while True:
            gen, coro = await self._queue.get()
            async with self._qsize_lock:
                self._qsize -= 1

            if coro is None:
                self._queue.task_done()
                logger.debug("[%s] Worker received stop sentinel.", self._name)
                break

            self._current_coro = coro
            self._current_coro_name = getattr(coro, "__qualname__", repr(coro))
            logger.debug(
                "[%s] Starting task: %s (gen=%d, remaining_qsize=%d)",
                self._name,
                self._current_coro_name,
                gen,
                self._qsize,
            )

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
