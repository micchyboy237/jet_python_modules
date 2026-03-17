import sys

from jet.audio.audio_waveform.speech_events import (
    SpeechSegmentEndEvent,
    SpeechSegmentStartEvent,
)
from jet.audio.audio_waveform.speech_handlers.base import SpeechSegmentHandler
from jet.audio.audio_waveform.speech_handlers.websocket_subtitle_sender import (
    SubtitleEntry,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class SubtitlePreviewWindow(QMainWindow):
    """Standalone window showing live .srt content"""

    def __init__(self, accumulator: SubtitleEntry):
        super().__init__()
        self.accumulator = accumulator

        self.setWindowTitle("Live Subtitles")
        self.resize(450, 550)

        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        # ✅ Top-right positioning
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - self.width() - 20, 20)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        # ✅ Top controls
        top_bar = QHBoxLayout()

        self.clear_btn = QPushButton("🗑")
        self.clear_btn.setToolTip("Clear all")
        self.clear_btn.clicked.connect(self.clear_all)

        top_bar.addWidget(self.clear_btn)
        top_bar.addStretch()

        layout.addLayout(top_bar)

        # ✅ Text area
        self.text_area = QTextBrowser()
        self.text_area.setReadOnly(True)
        self.text_area.setAcceptRichText(True)
        self.text_area.setStyleSheet("""
        QTextEdit {
            background-color: #0d1117;
            color: #c9d1d9;
            font-family: Consolas, 'Courier New', monospace;
            font-size: 11px;
            line-height: 1.25;
            padding: 6px;
            border: 1px solid #30363d;
        }
        """)

        # Handle link clicks (copy)
        self.text_area.setOpenExternalLinks(False)
        self.text_area.anchorClicked.connect(self._handle_anchor_click)

        layout.addWidget(self.text_area)
        self.setCentralWidget(central)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(800)

        # Track manual scrolling to suppress auto-scroll only if user not at bottom
        self._auto_scroll_enabled = True
        self.text_area.verticalScrollBar().valueChanged.connect(self._on_scroll)

        self.show()

    def clear_all(self):
        self.accumulator.entries.clear()
        self.text_area.clear()

    def _format_entry(self, i: int, e: dict) -> str:
        start = f"{e['start']:.2f}s"
        end = f"{e['end']:.2f}s"
        duration = f"{(e['end'] - e['start']):.2f}s"

        text = f"{e['ja']}\n{e['en']}".strip()
        if not text:
            text = "[no transcription]"

        return f"""
<div style="margin-bottom:6px;">
<b style="font-size:10px;">{i}</b>
<span style="font-size:10px; color:#8b949e;">
[{start} → {end}] ({duration})
</span>
<a href="copy:{i}" style="color:#58a6ff; text-decoration:none;">📋</a>
<br/>
<pre style="white-space:pre-wrap; margin:0; font-size:10px;">{text}</pre>
</div>
<hr style="border:none; border-top:1px solid #30363d; margin:4px 0;">
"""

    def _on_scroll(self):
        scrollbar = self.text_area.verticalScrollBar()
        # If user is near bottom → keep auto-scroll enabled
        if scrollbar.value() >= scrollbar.maximum() - 20:
            self._auto_scroll_enabled = True
        else:
            self._auto_scroll_enabled = False

    def _handle_anchor_click(self, url):
        if url.toString().startswith("copy:"):
            idx = int(url.toString().split(":")[1]) - 1
            if 0 <= idx < len(self.accumulator.entries):
                e = self.accumulator.entries[idx]
                text = f"{e['ja']}\n{e['en']}".strip()
                QApplication.clipboard().setText(text)
                # lightweight success feedback
                self.setWindowTitle("Copied ✓")
                QTimer.singleShot(800, lambda: self.setWindowTitle("Live Subtitles"))

    def update_display(self):
        if not self.accumulator.entries:
            self.text_area.setHtml("Waiting for transcribed segments…")
            return

        html = ""
        for i, e in enumerate(self.accumulator.entries, 1):
            html += self._format_entry(i, e)

        self.text_area.setHtml(html)

        if self._auto_scroll_enabled:
            cursor = self.text_area.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.text_area.setTextCursor(cursor)

    def closeEvent(self, event):
        # Allow real close when app is shutting down
        event.accept()


class LiveSrtPreviewHandler(SpeechSegmentHandler):
    def __init__(self, accumulator: SubtitleEntry):
        self.accumulator = accumulator
        self.preview_window: SubtitlePreviewWindow | None = None

        if QApplication.instance() is None:
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()

        self.preview_window = SubtitlePreviewWindow(self.accumulator)

    def on_segment_start(self, event: SpeechSegmentStartEvent) -> None:
        pass

    def on_segment_end(self, event: SpeechSegmentEndEvent) -> None:
        pass

    def close(self):
        if self.preview_window:
            self.preview_window.close()
