import subprocess
import sys

from jet.audio.audio_waveform.helpers.subtitle_entry import SubtitleEntry
from jet.audio.audio_waveform.speech_events import (
    SpeechSegmentEndEvent,
    SpeechSegmentStartEvent,
)
from jet.audio.audio_waveform.speech_handlers.base import SpeechSegmentHandler
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

    def __init__(self, accumulator: SubtitleEntry, show_ja_text: bool = False):
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

        # ✅ Toggle Japanese visibility
        self.hide_ja_btn = QPushButton("🇯🇵")
        self.hide_ja_btn.setToolTip("Toggle Japanese text")
        self.hide_ja_btn.setCheckable(True)
        self.hide_ja_btn.clicked.connect(self.toggle_hide_japanese)

        # Initialize visibility based on show_ja_text
        self.hide_japanese: bool = not show_ja_text
        self.hide_ja_btn.setChecked(self.hide_japanese)

        top_bar.addWidget(self.clear_btn)
        top_bar.addWidget(self.hide_ja_btn)
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
        self.text_area.setOpenLinks(False)
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

        self._last_html: str | None = None

    def clear_all(self):
        # Clear in-memory entries and UI
        self.accumulator.clear()
        self.text_area.clear()

    def toggle_hide_japanese(self):
        self.hide_japanese = self.hide_ja_btn.isChecked()
        # Force refresh
        self._last_html = None
        self.update_display()

    def _get_entry_text(self, e: dict) -> str:
        ja = e.get("ja", "").strip()
        en = e.get("en", "").strip()

        if self.hide_japanese:
            text = en
        else:
            text = f"{ja}\n{en}".strip()

        return text if text else "[no transcription]"

    def _format_entry(self, i: int, e: dict) -> str:
        # Compute gap from previous segment end
        prev_end = None
        if i > 1:
            prev = self.accumulator.entries[i - 2]
            prev_end = prev["end"]

        gap = e["start"] - prev_end if prev_end is not None else 0
        gap_str = f"{gap:.2f}s"
        duration = f"{(e['end'] - e['start']):.2f}s"
        text = self._get_entry_text(e)
        trigger_reason = e.get("trigger_reason") or "unknown"
        segment_dir = e.get("segment_dir")

        # New fields required by the task (now stored in entry via e.update(others))
        transcribed_pctg = e.get("transcribed_duration_pctg")
        coverage_label = e.get("coverage_label", "")
        pctg_str = (
            f"{float(transcribed_pctg):.1f}%"
            if isinstance(transcribed_pctg, (int, float))
            else "N/A"
        )
        cov_str = coverage_label or "N/A"

        open_link = (
            f'<a href="open:{i}" style="color:#58a6ff; text-decoration:none;">📂</a>'
            if segment_dir
            else ""
        )
        return f"""
<div style="margin-bottom:6px;">
<b style="font-size:10px;">{i}</b>
<span style="font-size:9px; color:#8b949e; line-height:1.1;">
[gap: {gap_str}] ({duration}) • <span style="color:#d2a8ff;">{trigger_reason}</span>
 • pctg: {pctg_str} • cov: <span style="color:#79c0ff;">{cov_str}</span>
</span>
<a href="copy:{i}" style="color:#58a6ff; text-decoration:none;">📋</a>
{open_link}
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

        elif url.toString().startswith("open:"):
            idx = int(url.toString().split(":")[1]) - 1
            if 0 <= idx < len(self.accumulator.entries):
                e = self.accumulator.entries[idx]
                segment_dir = e.get("segment_dir")
                if segment_dir:
                    try:
                        subprocess.Popen(["open", segment_dir])
                    except Exception:
                        pass

    def update_display(self):
        if not self.accumulator.entries:
            html = "Waiting for transcribed segments…"
        else:
            html_parts = []
            for i, e in enumerate(self.accumulator.entries, 1):
                html_parts.append(self._format_entry(i, e))
            html = "".join(html_parts)

        # ✅ Skip if nothing changed → prevents flicker
        if html == self._last_html:
            return

        self._last_html = html

        scrollbar = self.text_area.verticalScrollBar()
        old_value = scrollbar.value()
        at_bottom = old_value >= scrollbar.maximum() - 20

        self.text_area.setHtml(html)

        # ✅ Restore scroll position
        if at_bottom:
            cursor = self.text_area.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.text_area.setTextCursor(cursor)
        else:
            scrollbar.setValue(old_value)

    def closeEvent(self, event):
        # Allow real close when app is shutting down
        event.accept()


class LiveSrtPreviewHandler(SpeechSegmentHandler):
    def __init__(self, accumulator: SubtitleEntry, show_ja_text: bool = False):
        self.accumulator = accumulator
        self.preview_window: SubtitlePreviewWindow | None = None

        if QApplication.instance() is None:
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()

        self.preview_window = SubtitlePreviewWindow(
            self.accumulator,
            show_ja_text=show_ja_text,
        )

    def on_segment_start(self, event: SpeechSegmentStartEvent) -> None:
        pass

    def on_segment_end(self, event: SpeechSegmentEndEvent) -> None:
        pass

    def close(self):
        if self.preview_window:
            self.preview_window.close()
