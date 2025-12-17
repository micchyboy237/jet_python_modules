# subtitle_overlay.py
# One-liner: overlay = SubtitleOverlay.create()
# Then:      overlay.add_message("Your text here")   ← exactly what you want

import sys
import signal
import logging
from threading import Thread
import time
from typing import Optional, Callable

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea
)
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QFont

from rich.logging import RichHandler


def _setup_logging():
    logger = logging.getLogger("SubtitleOverlay")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True, markup=True, show_path=False)
        handler.setFormatter(logging.Formatter("[bold magenta]Subtitle[/] → %(message)s"))
        logger.addHandler(handler)
    return logger


class _Signals(QObject):
    _add_message = pyqtSignal(str)
    _clear = pyqtSignal()
    _toggle_minimize = pyqtSignal()


class SubtitleOverlay(QWidget):
    """
    Beautiful, thread-safe, reusable live subtitle overlay.

    Usage:
        overlay = SubtitleOverlay.create()
        overlay.add_message("Hello world!")           # ← your dream API
        overlay.add_message("Another line")
        overlay.clear()
    """

    def __init__(self, parent=None, title: Optional[str] = None):
        super().__init__(parent)
        self.logger = _setup_logging()
        self.signals = _Signals()
        self.history = []
        self.title = title

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 190); border-radius: 16px;")
        self.setFixedSize(960, 600)

        self._drag_pos = QPoint()
        self._build_ui()
        self._connect_signals()

        self.add_message("Live Subtitle Overlay • Ready")
        self.logger.info("[green]Ready – use .add_message('text')[/]")

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(16, 16, 16, 16)
        main.setSpacing(10)

        # === TOP CONTROL BAR (always visible) ===
        self.control_bar = QHBoxLayout()
        self.control_bar.setContentsMargins(12, 10, 12, 10)
        self.control_bar.setSpacing(12)

        # Optional title label
        if self.title:
            self.title_label = QLabel(self.title)
            self.title_label.setStyleSheet("color: #ffffff; font-size: 18px; font-weight: bold;")
            self.control_bar.addWidget(self.title_label)

        # Status indicator (LIVE)
        self.status_label = QLabel("LIVE")
        self.status_label.setStyleSheet("""
            color: #ff4444;
            background: rgba(255, 50, 50, 0.25);
            border: 1px solid rgba(255, 100, 100, 0.4);
            border-radius: 10px;
            padding: 4px 14px;
            font-weight: bold;
            font-size: 15px;
        """)
        self.control_bar.addWidget(self.status_label)
        self.control_bar.addStretch()

        # Minimize button
        self.min_btn = QPushButton("Minimize")
        self.min_btn.setFixedSize(38, 38)
        self.min_btn.setStyleSheet("""
            QPushButton {
                background: rgba(100, 180, 255, 0.25);
                color: white;
                border-radius: 19px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover { background: rgba(100, 180, 255, 0.5); }
        """)

        # Close button
        close_btn = QPushButton("×")
        close_btn.setFixedSize(38, 38)
        close_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 80, 80, 0.7);
                color: white;
                border-radius: 19px;
                font-weight: bold;
                font-size: 18px;
            }
            QPushButton:hover { background: rgba(255, 50, 50, 1); }
        """)
        close_btn.clicked.connect(QApplication.quit)

        self.control_bar.addWidget(self.min_btn)
        self.control_bar.addWidget(close_btn)

        # Optional: Make whole control bar draggable
        self.control_bar_widget = QWidget()
        self.control_bar_widget.setLayout(self.control_bar)
        self.control_bar_widget.setCursor(Qt.CursorShape.OpenHandCursor)
        main.addWidget(self.control_bar_widget)

        # === CONTENT AREA (scroll + summary) ===
        self.content_area = QVBoxLayout()
        self.content_area.setSpacing(8)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.content_layout.setSpacing(10)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.scroll.setWidget(self.content)

        self.summary = QLabel("0 lines • 0 chars")
        self.summary.setStyleSheet("""
            color: #aaffaa; font-size: 15px; padding: 8px;
            background: rgba(0, 100, 0, 0.5); border-radius: 8px;
        """)
        self.summary.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.content_area.addWidget(self.scroll, stretch=1)
        self.content_area.addWidget(self.summary)

        main.addLayout(self.content_area, stretch=1)

        # Initially full view
        self.min_btn.clicked.connect(self.toggle_minimize)
        self._is_minimized = False
        self._original_size = None

    def _connect_signals(self):
        self.signals._add_message.connect(self._do_add_message)
        self.signals._clear.connect(self.clear)
        self.signals._toggle_minimize.connect(self.toggle_minimize)

    def _center_window(self):
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.center().x() - self.width() // 2,
                  screen.center().y() - self.height() // 2)

    # ─────────────────────── Your Dream Public API ───────────────────────
    def add_message(self, text: str):
        """Call this from anywhere, any thread — 100% safe and clean."""
        if not text or not str(text).strip():
            return
        self.signals._add_message.emit(str(text).strip())

    def clear(self):
        """Clear all messages"""
        self.signals._clear.emit()

    def toggle_minimize(self):
        """Toggle between full transcript view and minimal floating bar"""
        if not self._is_minimized:
            # → Minimize
            self._original_size = self.size()
            self.setFixedHeight(64)
            self.content_area.layout().setEnabled(False)
            for i in reversed(range(self.content_area.layout().count())):
                item = self.content_area.layout().itemAt(i)
                if item.widget():
                    item.widget().hide()

            self.control_bar.layout().setContentsMargins(16, 12, 16, 12)
            self.status_label.setText("LIVE • MINIMIZED")
            if hasattr(self, 'title_label'):
                self.title_label.show()  # always keep title visible
            self.min_btn.setText("Restore")
            self._is_minimized = True
        else:
            # → Restore
            if self._original_size:
                self.setFixedSize(self._original_size)
            else:
                self.setFixedHeight(600)

            for i in range(self.content_area.layout().count()):
                item = self.content_area.layout().itemAt(i)
                if item.widget():
                    item.widget().show()
            self.content_area.layout().setEnabled(True)

            self.control_bar.layout().setContentsMargins(12, 10, 12, 10)
            self.status_label.setText("LIVE")
            self.min_btn.setText("Minimize")
            self._is_minimized = False

    def keyPressEvent(self, event):
        """Global hotkeys: Ctrl+M = toggle minimize, Ctrl+Q/Esc = quit"""
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if event.key() == Qt.Key.Key_M:
                self.toggle_minimize()
                return
            elif event.key() == Qt.Key.Key_Q:
                QApplication.quit()
                return
        elif event.key() == Qt.Key.Key_Escape:
            QApplication.quit()
            return
        super().keyPressEvent(event)

    # ─────────────────────── Internal (thread-safe) implementation ───────────────────────
    def _do_add_message(self, text: str):
        self.history.append(text)

        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: white; padding: 4px;")
        lbl.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 24))

        self.content_layout.addWidget(lbl)

        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()))

        total_chars = sum(len(line) for line in self.history)
        self.summary.setText(f"{len(self.history)} lines • {total_chars:,} chars")

    # Dragging
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_pos)
            e.accept()

    # ─────────────────────── One-liner factory ───────────────────────
    @classmethod
    def create(cls, app: Optional[QApplication] = None, title: Optional[str] = None) -> 'SubtitleOverlay':
        """
        One-liner to get a perfectly centered, always-on-top, live subtitle overlay.
        
        Features (all automatic):
        - Creates QApplication if needed
        - Prevents quit on close/minimize
        - Perfect centering on Windows/macOS/Linux
        - Graceful Ctrl+C shutdown
        - Thread-safe .add_message()
        - Optional custom title displayed in the top control bar
        """
        app = app or QApplication.instance() or QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(False)

        def _quit_on_sigint(sig, frame):
            app.quit()
        signal.signal(signal.SIGINT, _quit_on_sigint)

        overlay = cls(title=title)
        overlay.show()
        overlay.raise_()
        overlay.activateWindow()

        def _do_center():
            overlay.updateGeometry()
            screen = app.primaryScreen().availableGeometry()
            overlay.move(
                screen.center().x() - overlay.width() // 2,
                screen.center().y() - overlay.height() // 2,
            )
            overlay.logger.info("[green]Overlay centered perfectly[/]")

        QTimer.singleShot(50, _do_center)

        return overlay


def run_in_background(target: Callable[[], None], *, daemon: bool = True) -> Thread:
    """
    Start a long-running or blocking function in a background thread.
    
    Usage:
        run_in_background(transcription_loop)  # fire and forget
        # or
        thread = run_in_background(transcription_loop, daemon=False)  # keep reference if needed
    
    Returns the Thread instance for advanced use cases (join, etc.).
    """
    thread = Thread(target=target, daemon=daemon)
    thread.start()
    return thread


# Demo when run directly
if __name__ == "__main__":
    overlay = SubtitleOverlay.create(title="My Live Transcription")

    def demo():
        msgs = [
            "overlay.add_message() works!",
            "Super clean API",
            "Thread-safe",
            "Perfect for live translation",
            "Works on Windows + macOS",
            "This is a very long message to test word wrapping and auto-scrolling behavior during real-time streaming...",
        ]
        for i, msg in enumerate(msgs * 20):
            time.sleep(1.5)
            overlay.add_message(msg)
            if i == 10:
                overlay.toggle_minimize()
                time.sleep(1.5)
                overlay.toggle_minimize()

    run_in_background(demo)  # ← clean, readable, reusable

    sys.exit(QApplication.exec())
