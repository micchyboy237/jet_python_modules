# subtitle_overlay.py
# One-liner: overlay = SubtitleOverlay.create()
# Then:      overlay.add_message("Your text here")   ← exactly what you want

import sys
import signal
import logging
from typing import Optional

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = _setup_logging()
        self.signals = _Signals()
        self.history = []

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
        self._center_window()

        self.add_message("Live Subtitle Overlay • Ready")
        self.logger.info("[green]Ready – use .add_message('text')[/]")

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(16, 16, 16, 16)
        main.setSpacing(12)

        # Control bar
        ctrl = QHBoxLayout()
        ctrl.addStretch()

        self.min_btn = QPushButton("−")
        self.min_btn.setFixedSize(36, 36)
        self.min_btn.setStyleSheet("background: rgba(255,255,255,0.2); color: white; border-radius: 18px; font-weight: bold;")
        self.min_btn.clicked.connect(lambda: self.signals._toggle_minimize.emit())

        close_btn = QPushButton("×")
        close_btn.setFixedSize(36, 36)
        close_btn.setStyleSheet("background: rgba(220,50,50,0.8); color: white; border-radius: 18px; font-weight: bold;")
        close_btn.clicked.connect(QApplication.quit)

        ctrl.addWidget(self.min_btn)
        ctrl.addWidget(close_btn)
        main.addLayout(ctrl)

        # Scroll area
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

        main.addWidget(self.scroll, stretch=1)

        # Summary
        self.summary = QLabel("1 line • 34 chars")
        self.summary.setStyleSheet("color: #bbffbb; font-size: 15px; padding: 8px; background: rgba(0,100,0,0.5); border-radius: 8px;")
        self.summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main.addWidget(self.summary)

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
        if self.height() > 100:
            self.setFixedHeight(80)
            self.min_btn.setText("Box")
        else:
            self.setFixedHeight(600)
            self.min_btn.setText("Minus")

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
    def create(cls, app: Optional[QApplication] = None) -> 'SubtitleOverlay':
        """Create and return a ready-to-use overlay instance."""
        if app is None:
            app = QApplication.instance() or QApplication(sys.argv)
            app.setQuitOnLastWindowClosed(False)

        signal.signal(signal.SIGINT, lambda s, f: app.quit())

        overlay = cls()
        overlay.show()
        overlay.raise_()
        overlay.activateWindow()
        return overlay


# Demo when run directly
if __name__ == "__main__":
    overlay = SubtitleOverlay.create()

    import time
    from threading import Thread

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
            time.sleep(2.5)
            overlay.add_message(msg)
            if i == 10:
                overlay.toggle_minimize()
                time.sleep(1.5)
                overlay.toggle_minimize()

    Thread(target=demo, daemon=True).start()
    sys.exit(QApplication.exec())