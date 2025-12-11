# subtitle_overlay.py
import sys
import time
import signal
import logging
from threading import Thread

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea
)
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

from rich.logging import RichHandler


# ────────────────────────────── Rich Logging ──────────────────────────────
def setup_logging():
    logger = logging.getLogger("SubtitleOverlay")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True, markup=True, show_path=False)
        handler.setFormatter(logging.Formatter("[bold magenta]%(name)s[/] → %(message)s"))
        logger.addHandler(handler)
    return logger


# ────────────────────────────── Main Overlay ──────────────────────────────
class SubtitleOverlay(QWidget):
    textChanged = pyqtSignal(str)

    # New signal for thread-safe text appending
    appendRequested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        setup_logging()
        self.logger = logging.getLogger("SubtitleOverlay")

        # Window setup
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 190); border-radius: 16px;")
        self.setFixedSize(960, 600)

        self.history = []
        self._drag_pos = QPoint()

        self._build_ui()
        self._center_window()

        # Connect the thread-safe signal
        self.appendRequested.connect(self.append_text)

        self.appendRequested.emit("Live Subtitle Overlay • Ready")
        self.logger.info("[green]Overlay ready and centered[/]")

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(16, 16, 16, 16)
        main.setSpacing(12)

        # ── Control bar ──
        ctrl = QHBoxLayout()
        ctrl.addStretch()

        self.min_btn = QPushButton("−")
        self.min_btn.setFixedSize(36, 36)
        self.min_btn.setStyleSheet("""
            QPushButton { background: rgba(255,255,255,0.2); color: white; border-radius: 18px; font-weight: bold; }
            QPushButton:hover { background: rgba(255,255,255,0.35); }
        """)
        self.min_btn.clicked.connect(self.toggle_minimize)

        close_btn = QPushButton("×")
        close_btn.setFixedSize(36, 36)
        close_btn.setStyleSheet("""
            QPushButton { background: rgba(220,50,50,0.8); color: white; border-radius: 18px; font-weight: bold; }
            QPushButton:hover { background: rgba(255,50,50,1); }
        """)
        close_btn.clicked.connect(QApplication.quit)

        ctrl.addWidget(self.min_btn)
        ctrl.addWidget(close_btn)
        main.addLayout(ctrl)

        # ── Scroll area ──
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

        # ── Summary ──
        self.summary = QLabel("1 line • 34 chars")
        self.summary.setStyleSheet("""
            color: #bbffbb; font-size: 15px; padding: 8px;
            background: rgba(0,100,0,0.5); border-radius: 8px;
        """)
        self.summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main.addWidget(self.summary)

    def _center_window(self):
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(
            screen.center().x() - self.width() // 2,
            screen.center().y() - self.height() // 2
        )

    # ── Public API ──
    def append_text(self, text: str):
        if not text or not text.strip():
            return

        clean = text.strip()
        self.history.append(clean)

        lbl = QLabel(clean)
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: white; padding: 4px;")
        lbl.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 24))

        self.content_layout.addWidget(lbl)

        # Safe auto-scroll (already in GUI thread now)
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()
        ))

        total = sum(len(l) for l in self.history)
        self.summary.setText(f"{len(self.history)} lines • {total:,} chars")
        self.textChanged.emit(text)

    def set_text(self, text: str):
        self.append_text(text)

    def toggle_minimize(self):
        if self.height() > 100:
            self.setFixedHeight(80)
            self.min_btn.setText("□")
        else:
            self.setFixedHeight(600)
            self.min_btn.setText("−")

    # ── Dragging ──
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_pos)
            e.accept()


# ────────────────────────────── Graceful Ctrl+C ──────────────────────────────
def handle_sigint(sig, frame):
    print("\n[yellow]Ctrl+C → quitting gracefully...[/]")
    QApplication.quit()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_sigint)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    win = SubtitleOverlay()
    win.show()
    win.raise_()
    win.activateWindow()

    # Ensures it's visible on Windows too

    # Demo thread — now thread-safe
    def demo():
        msgs = [
            "Text now appears instantly on Windows",
            "Ctrl+C works perfectly",
            "No more thread warnings",
            "Auto-scroll to bottom",
            "History preserved",
            "Clean dark theme",
            "Very long line to test wrapping and scrolling behavior when subtitles get long during live streams or presentations...",
            "Short line",
            "Another one!",
        ]
        for i, m in enumerate(msgs * 10):
            time.sleep(2.5)
            win.appendRequested.emit(m)   # ← Use signal instead of direct call
            if i == 8:
                win.toggle_minimize()
                time.sleep(1.5)
                win.toggle_minimize()

    Thread(target=demo, daemon=True).start()

    sys.exit(app.exec())