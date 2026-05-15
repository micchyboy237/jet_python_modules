# jet.audio.speech_handlers.subtitle_overlay

"""
SubtitleOverlay
===============
A frameless, always-on-top, transparent PyQt6 window that displays
live subtitles (Japanese + English) received from the WebSocket server.

It also implements SpeechSegmentHandler so it can be declared alongside
WebsocketSubtitleSender in the handlers list, even though the actual
subtitle text arrives via the observer signal rather than on_segment_end.

Architecture
------------

  WS bg thread
      │
      ▼  on_subtitle_response(response)
  SubtitleResponseNotifier          ← QObject, emits subtitle_received signal
      │
      │  subtitle_received (Qt signal — thread-safe, queued across threads)
      ▼
  SubtitleOverlay.on_subtitle_received(response_dict)   ← runs on Qt main thread
      │
      ├─ updates _SubtitleWidget (Japanese + English labels)
      ├─ adjustSize() + repositions window at bottom-centre of screen
      ├─ show() / raise_()
      └─ resets auto-hide QTimer

Typical wiring (in main)
------------------------
    app = QApplication(sys.argv)
    overlay = SubtitleOverlay()
    notifier = SubtitleResponseNotifier()
    notifier.subtitle_received.connect(overlay.on_subtitle_received)
    sender.add_observer(notifier)
    overlay.show()
    sys.exit(app.exec())

Tunables
--------
All visual constants are module-level and easy to adjust without touching
the class logic.
"""

from __future__ import annotations

from abc import ABCMeta

from jet.audio.speech_handlers.base import SpeechSegmentHandler
from jet.audio.speech_handlers.speech_events import (
    SpeechSegmentEndEvent,
    SpeechSegmentStartEvent,
)
from jet.audio.speech_handlers.subtitle_observer import SubtitleResponseNotifier
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

# ── visual tunables ───────────────────────────────────────────────────────────
_HIDE_AFTER_MS: int = 6_000  # auto-hide N ms after the last subtitle update
_WINDOW_WIDTH: int = 900  # max overlay width (px); clamped to screen width
_WINDOW_OPACITY: float = 0.92  # overall window transparency (0.0 = invisible)
_BG_RGBA: tuple[int, int, int, int] = (18, 18, 20, 215)  # near-black background
_JA_COLOR: str = "#FFFFFF"  # Japanese text colour
_EN_COLOR: str = "#B0C8FF"  # English text colour (soft blue)
_JA_FONT_SIZE: int = 22  # Japanese label point size
_EN_FONT_SIZE: int = 16  # English label point size
_CORNER_RADIUS: int = 14  # background rounded-rect corner radius
_BOTTOM_MARGIN: int = 50  # px from screen bottom edge


# ── inner widget ──────────────────────────────────────────────────────────────


class _SubtitleWidget(QWidget):
    """
    Draws a rounded-rect background and hosts the two text labels.

    Kept as a private inner class so SubtitleOverlay has a single
    content child to manage.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._build_layout()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 16, 28, 16)
        layout.setSpacing(8)

        self._ja_label = self._make_label(_JA_COLOR, _JA_FONT_SIZE, bold=True)
        self._en_label = self._make_label(_EN_COLOR, _EN_FONT_SIZE, bold=False)

        layout.addWidget(self._ja_label)
        layout.addWidget(self._en_label)

    @staticmethod
    def _make_label(color: str, pt_size: int, *, bold: bool) -> QLabel:
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        font = QFont()
        font.setPointSize(pt_size)
        font.setBold(bold)
        label.setFont(font)
        label.setStyleSheet(f"color: {color}; background: transparent;")
        return label

    # ── painting ───────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        """Draw a semi-transparent rounded rectangle as the subtitle background."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(
            0.0,
            0.0,
            float(self.width()),
            float(self.height()),
            _CORNER_RADIUS,
            _CORNER_RADIUS,
        )
        r, g, b, a = _BG_RGBA
        painter.fillPath(path, QColor(r, g, b, a))

    # ── public API ─────────────────────────────────────────────────────────────

    def set_subtitles(self, ja: str, en: str) -> None:
        """Update both lines. Hides the English label when empty."""
        self._ja_label.setText(ja)
        self._en_label.setText(en)
        self._en_label.setVisible(bool(en))

    def clear(self) -> None:
        self._ja_label.clear()
        self._en_label.clear()


# ── overlay window ────────────────────────────────────────────────────────────


# PyQt6 uses its own internal metaclass (sip.wrappertype).
# ABCMeta is the metaclass of SpeechSegmentHandler (via ABC).
# Python raises TypeError if a class inherits from both without a unified metaclass.
# The fix: create a merged metaclass that subclasses both.
class _QtABCMeta(type(QWidget), ABCMeta):
    pass


class SubtitleOverlay(QWidget, SpeechSegmentHandler, metaclass=_QtABCMeta):
    """
    Frameless, always-on-top, transparent subtitle window.

    Implements SpeechSegmentHandler so it can appear in the handlers list,
    but subtitle updates arrive via on_subtitle_received (a Qt slot), not
    via on_segment_end. Pair this with SubtitleResponseNotifier to connect
    WebsocketSubtitleSender to this overlay.

    Thread safety
    -------------
    on_subtitle_received is a Qt slot — it always runs on the Qt main thread
    when connected to a SubtitleResponseNotifier signal. Do not call it
    directly from background threads.
    """

    def __init__(
        self,
        hide_after_ms: int = _HIDE_AFTER_MS,
        parent: QWidget | None = None,
    ) -> None:
        # QWidget must be initialised first (MRO: QWidget before SpeechSegmentHandler)
        QWidget.__init__(self, parent)
        self._hide_after_ms = hide_after_ms
        self._setup_window()
        self._setup_content()
        self._setup_timer()

    # ── initialisation ─────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool  # keeps window off the taskbar
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowOpacity(_WINDOW_OPACITY)
        self._position_at_bottom()

    def _position_at_bottom(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geom = screen.availableGeometry()
        w = min(_WINDOW_WIDTH, geom.width() - 40)
        # Initial height estimate; adjusted after first subtitle arrives.
        self.setGeometry(
            (geom.width() - w) // 2,
            geom.height() - 160,
            w,
            110,
        )

    def _setup_content(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._subtitle_widget = _SubtitleWidget(self)
        layout.addWidget(self._subtitle_widget)

    def _setup_timer(self) -> None:
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._on_hide_timeout)

    # ── timer ──────────────────────────────────────────────────────────────────

    def _on_hide_timeout(self) -> None:
        """Called on the Qt main thread when the auto-hide timer fires."""
        self._subtitle_widget.clear()
        self.hide()

    # ── Qt slot ────────────────────────────────────────────────────────────────

    def on_subtitle_received(self, response: dict) -> None:
        """
        Qt slot — always runs on the Qt main thread (via queued signal).

        Extracts Japanese and English text from the response dict and
        updates the overlay. Resets the auto-hide timer on every call.
        """
        ja = response.get("transcription_ja", "").strip()
        en = response.get("translation_en", "").strip()

        if not ja and not en:
            return

        self._subtitle_widget.set_subtitles(ja, en)
        self.adjustSize()
        self._reposition()
        self.show()
        self.raise_()

        # Restart the timer so the overlay stays visible for _hide_after_ms
        # from the *last* subtitle update.
        self._hide_timer.start(self._hide_after_ms)

    def _reposition(self) -> None:
        """
        Re-centre the overlay horizontally and pin it to the screen bottom.
        Called after adjustSize() so height is up-to-date.
        """
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geom = screen.availableGeometry()
        w = min(_WINDOW_WIDTH, geom.width() - 40)
        x = (geom.width() - w) // 2
        y = geom.height() - self.height() - _BOTTOM_MARGIN
        self.setGeometry(x, y, w, self.height())

    # ── SpeechSegmentHandler (required interface) ──────────────────────────────

    def on_segment_start(self, event: SpeechSegmentStartEvent) -> None:
        """No-op. The overlay reacts to WS responses, not raw segment boundaries."""
        pass

    def on_segment_end(self, event: SpeechSegmentEndEvent) -> None:
        """No-op. Subtitle text comes via the observer signal, not the audio segment."""
        pass

    # ── convenience factory ────────────────────────────────────────────────────

    @classmethod
    def create_and_connect(
        cls,
        sender,  # WebsocketSubtitleSender
        hide_after_ms: int = _HIDE_AFTER_MS,
    ) -> "SubtitleOverlay":
        """
        Factory: create an overlay, wire it to sender, return it ready to show.

        Example
        -------
            app = QApplication(sys.argv)
            sender = WebsocketSubtitleSender(...)
            overlay = SubtitleOverlay.create_and_connect(sender)
            overlay.show()
            sys.exit(app.exec())
        """
        overlay = cls(hide_after_ms=hide_after_ms)
        notifier = SubtitleResponseNotifier(parent=overlay)
        notifier.subtitle_received.connect(overlay.on_subtitle_received)
        sender.add_observer(notifier)
        return overlay
