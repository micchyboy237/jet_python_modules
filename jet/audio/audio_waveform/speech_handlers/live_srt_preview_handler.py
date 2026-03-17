# jet.audio.audio_waveform.speech_handlers.live_srt_preview_handler

import sys

from jet.audio.audio_waveform.speech_events import (
    SpeechSegmentEndEvent,
    SpeechSegmentStartEvent,
)
from jet.audio.audio_waveform.speech_handlers.base import SpeechSegmentHandler
from jet.audio.audio_waveform.speech_handlers.websocket_subtitle_sender import (
    SubtitleEntry,
)
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget


class SubtitlePreviewWindow(QMainWindow):
    """Standalone window showing live .srt content"""

    def __init__(self, accumulator: SubtitleEntry):
        super().__init__()
        self.accumulator = accumulator

        self.setWindowTitle("Live Japanese → English Subtitles (real-time)")
        self.resize(780, 460)
        self.move(120, 120)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setAcceptRichText(False)
        self.text_area.setStyleSheet("""
            QTextEdit {
                background-color: #0d1117;
                color: #c9d1d9;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 15px;
                line-height: 1.45;
                padding: 10px;
                border: 1px solid #30363d;
            }
        """)
        layout.addWidget(self.text_area)

        self.setCentralWidget(central)

        # Update timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(1400)  # ~every 1.4 seconds

        self.show()

    def update_display(self):
        srt_content = self.accumulator.to_srt()
        display_text = (
            srt_content if srt_content.strip() else "Waiting for transcribed segments…"
        )
        self.text_area.setPlainText(display_text)

        # Auto-scroll to bottom
        sb = self.text_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        # Optional: keep running in background or minimize instead of close
        event.ignore()
        self.hide()


class LiveSrtPreviewHandler(SpeechSegmentHandler):
    """
    Creates and manages a real-time .srt preview window using PyQt6.
    Also handles final .srt save on exit.
    """

    def __init__(self, accumulator: SubtitleEntry):
        self.accumulator = accumulator
        self.preview_window: SubtitlePreviewWindow | None = None

        # Make sure QApplication exists
        if QApplication.instance() is None:
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()

        # Create preview in main thread
        self.preview_window = SubtitlePreviewWindow(self.accumulator)

    def on_segment_start(self, event: SpeechSegmentStartEvent) -> None:
        pass  # we update on end

    def on_segment_end(self, event: SpeechSegmentEndEvent) -> None:
        # Just trigger UI update — timer is already running
        if self.preview_window and self.preview_window.isVisible():
            # Can force immediate update if you want (optional)
            # self.preview_window.update_display()
            pass

    def close(self):
        if self.preview_window:
            self.preview_window.close()
