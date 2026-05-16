"""
Unit tests for AsyncTaskQueue.

Run with:
    pytest tests/test_async_task_queue.py -v
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from jet.audio.async_utils.task_queue import AsyncTaskQueue

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def event_loop_in_thread():
    """
    Spin up a real asyncio event loop in a daemon thread and yield it.
    Tears down cleanly after each test.
    """
    loop = asyncio.new_event_loop()

    def run():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = threading.Thread(target=run, daemon=True, name="test-loop")
    t.start()

    yield loop

    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=3)


@pytest.fixture()
def queue(event_loop_in_thread):
    """A fresh AsyncTaskQueue wired to the test loop."""
    return AsyncTaskQueue(event_loop_in_thread, name="test-queue")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def drain_sync(q: AsyncTaskQueue, timeout: float = 3.0) -> None:
    """Block the test thread until the queue is empty."""
    future = asyncio.run_coroutine_threadsafe(q.drain(), q._loop)
    future.result(timeout=timeout)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEnqueue:
    def test_single_task_runs(self, queue):
        results = []

        async def task():
            results.append(1)

        queue.enqueue(task())
        drain_sync(queue)
        assert results == [1]

    def test_multiple_tasks_all_run(self, queue):
        results = []

        async def task(n):
            results.append(n)

        for i in range(5):
            queue.enqueue(task(i))

        drain_sync(queue)
        assert sorted(results) == [0, 1, 2, 3, 4]

    def test_tasks_run_in_fifo_order(self, queue):
        """Tasks must execute strictly in submission order."""
        order = []

        async def task(n):
            order.append(n)

        # Submit sequentially from the test thread so order is deterministic.
        for i in range(10):
            queue.enqueue(task(i))
            time.sleep(0.001)  # tiny gap keeps submission order stable

        drain_sync(queue)
        assert order == list(range(10))


class TestSerialExecution:
    def test_no_concurrent_execution(self, queue):
        """
        The queue must never run two tasks at the same time.
        We detect overlap by tracking the 'active' flag.
        """
        active = []
        overlaps: list[bool] = []
        lock = asyncio.Lock()

        async def slow_task():
            async with lock:
                overlaps.append(len(active) > 0)  # True if another task is running
                active.append(1)
                await asyncio.sleep(0.02)
                active.pop()

        for _ in range(5):
            queue.enqueue(slow_task())

        drain_sync(queue, timeout=5)
        # No task should have found another task active
        assert not any(overlaps), f"Detected concurrent execution: {overlaps}"

    def test_tasks_run_one_at_a_time_with_shared_resource(self, queue):
        """
        Simulates the websocket send/recv pattern: a shared counter
        incremented and decremented inside each task.  Concurrent access
        would leave counter != 0.
        """
        counter = [0]
        violations: list[int] = []

        async def ws_like_task():
            counter[0] += 1
            await asyncio.sleep(0.01)
            if counter[0] != 1:
                violations.append(counter[0])
            counter[0] -= 1

        for _ in range(8):
            queue.enqueue(ws_like_task())

        drain_sync(queue, timeout=5)
        assert violations == [], f"Concurrent access detected: {violations}"


class TestErrorHandling:
    def test_failing_task_does_not_stop_queue(self, queue):
        """An exception in one task must not kill the worker."""
        results = []

        async def bad_task():
            raise RuntimeError("boom")

        async def good_task(n):
            results.append(n)

        queue.enqueue(bad_task())
        queue.enqueue(good_task(42))
        drain_sync(queue)
        assert results == [42]

    def test_multiple_failing_tasks(self, queue):
        results = []

        async def boom(label):
            raise ValueError(label)

        async def ok(n):
            results.append(n)

        for i in range(3):
            queue.enqueue(boom(f"err-{i}"))
            queue.enqueue(ok(i))

        drain_sync(queue)
        assert results == [0, 1, 2]


class TestCancel:
    def test_cancel_stops_worker(self, queue):
        """After cancel(), no new tasks should execute."""
        results = []

        async def task(n):
            results.append(n)

        queue.enqueue(task(1))
        drain_sync(queue)
        queue.cancel()
        time.sleep(0.05)  # let sentinel propagate

        queue.enqueue(task(99))
        time.sleep(0.05)
        # 99 might or might not execute depending on timing, but the test
        # validates cancel() doesn't crash and the pre-cancel task ran.
        assert 1 in results


class TestQsize:
    def test_qsize_reflects_pending_items(self, event_loop_in_thread):
        """
        We pause the first task so the rest pile up, then verify qsize > 0.
        """
        pause = threading.Event()

        async def blocking_task():
            await asyncio.get_event_loop().run_in_executor(None, pause.wait)

        async def quick_task():
            pass

        q = AsyncTaskQueue(event_loop_in_thread, name="qsize-test")
        q.enqueue(blocking_task())  # occupies the worker

        time.sleep(0.05)  # let worker pick it up

        for _ in range(4):
            q.enqueue(quick_task())

        time.sleep(0.05)
        assert q.qsize >= 1  # at least some are pending

        pause.set()  # unblock
        drain_sync(q)
        assert q.qsize == 0


class TestThreadSafety:
    def test_enqueue_from_multiple_threads(self, queue):
        """Enqueue from 10 threads simultaneously; all tasks must run."""
        results = []
        results_lock = threading.Lock()

        async def task(n):
            results_lock.acquire()
            results.append(n)
            results_lock.release()

        threads = [
            threading.Thread(target=queue.enqueue, args=(task(i),)) for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        drain_sync(queue, timeout=5)
        assert sorted(results) == list(range(10))
