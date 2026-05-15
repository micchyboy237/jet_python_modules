# jet.audio.speech_handlers.subtitle_observer

"""
SubtitleResponseNotifier
========================
A thin QObject that bridges the asyncio WS background thread
and the Qt main thread via a thread-safe Qt signal.

How it works
------------
PyQt6 automatically queues signal emissions that cross thread boundaries.
This means on_subtitle_response() can be called from the asyncio WS loop
thread without any explicit locking — Qt will safely deliver the signal
to slots connected on the main thread.

Usage
-----
    notifier = SubtitleResponseNotifier()
    notifier.subtitle_received.connect(overlay.on_subtitle_received)
    sender.add_observer(notifier)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jet.audio.speech_handlers.api_types import ServerResponse
from PyQt6.QtCore import QObject, pyqtSignal


@runtime_checkable
class SubtitleObserver(Protocol):
    """
    Structural protocol for any object that wants subtitle responses.

    Implementations are called from the asyncio WS background thread,
    so they must be thread-safe. The recommended approach is to use a
    SubtitleResponseNotifier, which converts the call into a queued Qt signal.
    """

    def on_subtitle_response(self, response: ServerResponse) -> None: ...


class SubtitleResponseNotifier(QObject):
    """
    Converts a raw on_subtitle_response() call (from any thread)
    into a queued Qt signal delivery (always on the Qt main thread).

    This is the canonical way to connect WebsocketSubtitleSender to any
    Qt widget without locks, polling, or explicit thread marshalling.
    """

    # Carries the full response dict. Connected slots always run on the Qt main thread.
    subtitle_received = pyqtSignal(dict)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

    # Satisfies the SubtitleObserver protocol.
    def on_subtitle_response(self, response: ServerResponse) -> None:
        """
        Called from the WS background thread.
        Emitting a signal across threads is thread-safe in Qt — Qt queues
        the delivery and dispatches it on the receiver's thread (main thread).
        """
        self.subtitle_received.emit(dict(response))
