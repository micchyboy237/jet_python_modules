"""
SubtitleOverlay  (FireRed VAD edition)
======================================
A PyQt6 always-on-top preview window that displays live subtitle entries
received from the WebSocket subtitle server.
Features
--------
- Rich HTML log of all segments: Japanese + English text, per-segment metadata
- Per-entry action links: copy 📋, open folder 📂, play audio ▶
- Toggle Japanese line visibility
- Clear all entries
- **NEW: Global Reset checkbox** — when checked, clearing also POSTs to
  /global/reset on the server
- Auto-scroll that follows new entries unless the user has scrolled up
- QSoundEffect for in-app WAV playback
- VAD badge: FireRed only (hard-coded; no registry)
- Metadata row: gap, duration, balanced VAD score, speech %, speech duration,
  transcription %, coverage label
- VADScorer row: composite score and quality label with colour coding
- Implements SpeechSegmentHandler (on_segment_end is a no-op;
  subtitle data arrives via SubtitleResponseNotifier Qt signal)
- **NEW: Queue status display** — shows currently processing segment and pending items
Observer wiring (done in main)
-------------------------------
    app = QApplication(sys.argv)
    overlay = SubtitleOverlay.create_and_connect(sender)
    overlay.show()
    sys.exit(app.exec())
"""

from __future__ import annotations

from abc import ABCMeta
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from jet.audio.speech_handlers.api_types import SubtitleNotification
from jet.audio.speech_handlers.base import SpeechSegmentHandler
from jet.audio.speech_handlers.queue_observer import QueueObserver
from jet.audio.speech_handlers.speech_events import (
    SpeechSegmentEndEvent,
    SpeechSegmentStartEvent,
)
from jet.audio.speech_handlers.subtitle_observer import SubtitleResponseNotifier
from jet.audio.speech_handlers.subtitle_overlay_actions import SubtitleOverlayActions
from jet.audio.speech_handlers.subtitle_overlay_log_viewer import (
    SubtitleOverlayLogViewer,
)
from jet.audio.speech_handlers.subtitle_overlay_style import (
    OVERLAY_HEIGHT,
    OVERLAY_WIDTH,
    _format_entry,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QTextCursor
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from rich.console import Console

console = Console()


class _QtABCMeta(type(QMainWindow), ABCMeta):
    pass


class SubtitleOverlay(
    QMainWindow,
    SpeechSegmentHandler,
    QueueObserver,
    SubtitleOverlayActions,  # added
    SubtitleOverlayLogViewer,  # added
    metaclass=_QtABCMeta,
):
    """
    Always-on-top subtitle preview window.
    NEW: Includes a "Global Reset on Clear" checkbox. When checked, pressing
    the clear button also calls POST /global/reset on the live-subtitles server.
    """

    _POLL_MS: int = 800
    _WINDOW_TITLE: str = "Live Subtitles"
    _TEXT_AREA_STYLE: str = """
        QTextEdit {
            background-color: #0d1117;
            color: #c9d1d9;
            font-family: Consolas, 'Courier New', monospace;
            font-size: 11px;
            line-height: 1.25;
            padding: 6px;
            border: 1px solid #30363d;
        }
    """
    queue_status_updated = pyqtSignal(str, int, str)

    @classmethod
    def create_and_connect(
        cls,
        sender,
        show_ja_text: bool = False,
        extra_clear_paths: Sequence[Path] = (),
        on_clear: Optional[callable] = None,
        global_reset_handler=None,
    ) -> "SubtitleOverlay":
        overlay = cls(
            show_ja_text=show_ja_text,
            extra_clear_paths=extra_clear_paths,
            on_clear=on_clear,
            global_reset_handler=global_reset_handler,
        )
        overlay.set_sender_reference(sender)
        sender.set_queue_observer(overlay)
        console.print("[debug][SubtitleOverlay] Registered as queue observer[/debug]")
        notifier = SubtitleResponseNotifier(parent=overlay)
        notifier.subtitle_received.connect(overlay.on_subtitle_received)
        sender.add_observer(notifier)
        return overlay

    def __init__(
        self,
        show_ja_text: bool = False,
        extra_clear_paths: Sequence[Path] = (),
        on_clear: Optional[callable] = None,
        global_reset_handler=None,
        parent: QWidget | None = None,
    ) -> None:
        QMainWindow.__init__(self, parent)
        self._entries: list[dict] = []
        self._hide_japanese: bool = not show_ja_text
        self._last_html: Optional[str] = None
        self._auto_scroll_enabled: bool = True
        self._extra_clear_paths: tuple[Path, ...] = tuple(extra_clear_paths)
        self._on_clear: Optional[callable] = on_clear
        self._global_reset_handler = global_reset_handler
        self._queue_status_text: str = "Idle"
        self._queue_pending_count: int = 0
        self._queue_status_color: str = "#8b949e"
        self._setup_window()
        self._setup_ui()
        self._setup_sound()
        self._setup_poll_timer()
        self.queue_status_updated.connect(self._update_queue_display)
        console.print(
            "[debug][SubtitleOverlay] Queue status display initialized[/debug]"
        )

    def _setup_window(self) -> None:
        self.setWindowTitle(self._WINDOW_TITLE)
        self.resize(OVERLAY_WIDTH, OVERLAY_HEIGHT)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            self.move(geom.right() - self.width() - 20, 20)

    def _setup_ui(self) -> None:
        from jet.audio.speech_handlers.settings_cache import AppSettingsStore

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        top_bar = QHBoxLayout()
        self._clear_btn = QPushButton("🗑")
        self._clear_btn.setToolTip("Clear all entries")
        self._clear_btn.clicked.connect(self._clear_all)
        self._clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 3px;
                color: #c9d1d9;
                font-size: 12px;
                padding: 3px 6px;
            }
            QPushButton:hover {
                background-color: #30363d;
            }
        """)
        self._hide_ja_btn = QPushButton("🇯🇵")
        self._hide_ja_btn.setToolTip("Toggle Japanese text")
        self._hide_ja_btn.setCheckable(True)
        self._hide_ja_btn.setChecked(self._hide_japanese)
        self._hide_ja_btn.clicked.connect(self._toggle_japanese)
        self._hide_ja_btn.setStyleSheet("""
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 3px;
                color: #c9d1d9;
                font-size: 12px;
                padding: 3px 6px;
            }
            QPushButton:hover {
                background-color: #30363d;
            }
            QPushButton:checked {
                background-color: #1f3b5c;
                border-color: #58a6ff;
            }
        """)
        self._settings_store = AppSettingsStore()
        self._lang_btn = QPushButton()
        self._lang_btn.setToolTip("Select transcription language")
        self._lang_btn.clicked.connect(self._cycle_language)
        self._lang_btn.setStyleSheet("""
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 3px;
                color: #c9d1d9;
                font-size: 11px;
                padding: 3px 6px;
            }
            QPushButton:hover {
                background-color: #30363d;
            }
        """)
        self._refresh_language_button()
        self._global_reset_checkbox = QCheckBox("🔄 Global Reset")
        self._global_reset_checkbox.setToolTip(
            "When checked, clearing also resets the server state via /global/reset"
        )
        self._global_reset_checkbox.setChecked(
            self._settings_store.global_reset_on_clear
        )
        self._global_reset_checkbox.stateChanged.connect(self._on_global_reset_toggled)
        self._global_reset_checkbox.setStyleSheet("""
            QCheckBox {
                color: #8b949e;
                font-size: 10px;
                spacing: 4px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #30363d;
                border-radius: 3px;
                background-color: #21262d;
            }
            QCheckBox::indicator:checked {
                background-color: #238636;
                border-color: #2ea043;
            }
        """)
        top_bar.addWidget(self._clear_btn)
        top_bar.addWidget(self._hide_ja_btn)
        top_bar.addWidget(self._lang_btn)
        top_bar.addWidget(self._global_reset_checkbox)
        top_bar.addStretch()
        layout.addLayout(top_bar)
        self._queue_status_bar = QHBoxLayout()
        self._queue_status_bar.setContentsMargins(0, 2, 0, 2)
        self._queue_status_bar.setSpacing(4)
        self._queue_label = QLabel("📡")
        self._queue_label.setStyleSheet(
            "color: #8b949e; font-size: 10px; padding: 0px 2px;"
        )
        self._queue_label.setToolTip("WebSocket queue status")
        self._queue_status_display = QLabel(self._queue_status_text)
        self._queue_status_display.setStyleSheet(
            f"color: {self._queue_status_color}; "
            "font-size: 10px; "
            "font-family: 'Consolas', 'Courier New', monospace; "
            "padding: 2px 4px; "
            "background-color: #161b22; "
            "border: 1px solid #21262d; "
            "border-radius: 3px;"
        )
        self._queue_status_display.setToolTip(
            "Current queue status\n"
            "📡 = WebSocket connection status\n"
            "⏳ = Currently processing\n"
            "📋 = Pending items in queue\n"
            "✓ = Idle (no pending work)\n"
            "Click 📜 for detailed history"
        )
        self._queue_pending_label = QLabel()
        self._queue_pending_label.setStyleSheet(
            "color: #58a6ff; "
            "font-size: 10px; "
            "font-family: 'Consolas', 'Courier New', monospace; "
            "font-weight: bold; "
            "padding: 2px 4px; "
            "background-color: #161b22; "
            "border: 1px solid #21262d; "
            "border-radius: 3px;"
        )
        self._queue_pending_label.setToolTip("Number of segments waiting to be sent")
        self._log_btn = QPushButton("📜")
        self._log_btn.setFixedSize(26, 20)
        self._log_btn.setToolTip(
            "View status log history\n"
            "Shows recent WebSocket events:\n"
            "• Segment send/receive\n"
            "• Retry attempts\n"
            "• Connection changes\n"
            "• Error details"
        )
        self._log_btn.clicked.connect(self._show_log_history)
        self._log_btn.setStyleSheet("""
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 3px;
                color: #8b949e;
                font-size: 10px;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #30363d;
                color: #c9d1d9;
                border-color: #58a6ff;
            }
            QPushButton:pressed {
                background-color: #1f3b5c;
            }
        """)
        self._update_queue_display()
        self._queue_status_bar.addWidget(self._queue_label)
        self._queue_status_bar.addWidget(self._queue_status_display, 1)
        self._queue_status_bar.addWidget(self._queue_pending_label)
        self._queue_status_bar.addWidget(self._log_btn)
        layout.addLayout(self._queue_status_bar)
        self._retry_progress = QProgressBar()
        self._retry_progress.setMaximumHeight(3)
        self._retry_progress.setTextVisible(False)
        self._retry_progress.setStyleSheet("""
            QProgressBar {
                background-color: #161b22;
                border: none;
                border-radius: 1px;
            }
            QProgressBar::chunk {
                background-color: #f0883e;
                border-radius: 1px;
            }
        """)
        self._retry_progress.setToolTip(
            "Retry progress — fills up with each retry attempt"
        )
        self._retry_progress.hide()
        layout.addWidget(self._retry_progress)
        self._text_area = QTextBrowser()
        self._text_area.setReadOnly(True)
        self._text_area.setAcceptRichText(True)
        self._text_area.setStyleSheet(self._TEXT_AREA_STYLE)
        self._text_area.setOpenExternalLinks(False)
        self._text_area.setOpenLinks(False)
        self._text_area.anchorClicked.connect(self._handle_anchor_click)
        self._text_area.verticalScrollBar().valueChanged.connect(self._on_scroll)
        self._text_area.setToolTip(
            "Live subtitle entries\n"
            "📋 Copy text | 📂 Open folder | ▶ Play audio\n"
            "VAD badges show speech detection quality"
        )
        layout.addWidget(self._text_area)
        self.setCentralWidget(central)

    def _on_global_reset_toggled(self, state: int) -> None:
        """Persist checkbox state whenever the user toggles it."""
        checked = state == Qt.CheckState.Checked.value
        self._settings_store.set_global_reset_on_clear(checked)
        console.print(
            f"[debug][SubtitleOverlay] Global Reset on Clear: {'ON' if checked else 'OFF'}[/debug]"
        )

    def _refresh_language_button(self) -> None:
        """Update the language button text to reflect current selection."""
        lang = self._settings_store.language
        label_map = {"auto": "🌐 Auto", "en": "🇺🇸 EN", "ja": "🇯🇵 JA"}
        self._lang_btn.setText(label_map.get(lang, f"🌐 {lang}"))

    def _cycle_language(self) -> None:
        """Cycle to the next language and update UI + sender."""
        new_lang = self._settings_store.cycle_language()
        self._refresh_language_button()
        if hasattr(self, "_sender") and self._sender is not None:
            self._sender.set_language(new_lang)

    @property
    def language(self) -> str:
        """Expose current language so external code can read it."""
        return self._settings_store.language

    def set_sender_reference(self, sender) -> None:
        """
        Store a reference to the WebsocketSubtitleSender so the overlay
        can update the sender's language when the user changes it.
        Called automatically by create_and_connect.
        """
        self._sender = sender

    def _setup_sound(self) -> None:
        self._sound_effect = QSoundEffect()

    def _setup_poll_timer(self) -> None:
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._refresh_display)
        self._poll_timer.start(self._POLL_MS)
        # Initialize mixin state used by SubtitleOverlayLogViewer
        self._segment_statuses: dict[int, dict] = {}
        self._segment_order: list[int] = []
        self._max_log_segments: int = 25
        self._log_viewer: Optional[QDialog] = None

    def on_subtitle_received(self, notification: SubtitleNotification) -> None:
        ja = notification.get("ja_text", "").strip()
        en = notification.get("en_text", "").strip()
        if not ja and not en:
            return
        vad_score: Optional[float] = notification.get("vad_score")
        speaker_label = notification.get("speaker_label", "")
        speaker_confidence = notification.get("speaker_confidence")
        speaker_match_type = notification.get("speaker_match_type", "")
        diarization = notification.get("diarization", {})
        segment_number = notification.get("num", 0)
        self._entries.append(
            {
                "ja": ja,
                "en": en,
                "segment_number": segment_number,
                "start": notification.get("start_sec", 0.0),
                "end": notification.get("end_sec", 0.0),
                "start_time_utc": notification.get("start_time_utc"),
                "end_time_utc": notification.get("end_time_utc"),
                "end_reason": notification.get("end_reason", "true_silence"),
                "vad_score": vad_score,
                "speech_frames_pctg": notification.get("speech_frames_pctg"),
                "speech_dur_sec": notification.get("speech_dur_sec"),
                "transcribed_duration_pctg": notification.get(
                    "transcribed_duration_pctg"
                ),
                "coverage_label": notification.get("coverage_label", ""),
                "segment_dir": notification.get("segment_dir"),
                "vad_composite_score": notification.get("vad_composite_score"),
                "vad_quality_label": notification.get("vad_quality_label"),
                "speaker_label": speaker_label,
                "speaker_confidence": speaker_confidence,
                "speaker_match_type": speaker_match_type,
                "diarization": diarization,
            }
        )
        self._last_html = None

    def _refresh_display(self) -> None:
        """Render entries sorted by segment number for consistent chronological order."""
        # Sort by segment number so retried segments appear in correct position
        sorted_entries = sorted(self._entries, key=lambda e: e.get("segment_number", 0))
        html_parts: list[str] = []
        prev_entry: Optional[dict] = None
        for i, entry in enumerate(sorted_entries, 1):
            ja = entry.get("ja", "").strip()
            en = entry.get("en", "").strip()
            text = en if self._hide_japanese else f"{ja}\n{en}".strip()
            if not text.strip():
                prev_entry = entry
                continue
            html_parts.append(_format_entry(i, entry, prev_entry, self._hide_japanese))
            prev_entry = entry
        html = (
            "".join(html_parts)
            if html_parts
            else "<span style='color:#8b949e;'>Waiting for transcribed segments…</span>"
        )
        if html == self._last_html:
            return
        self._last_html = html
        scrollbar = self._text_area.verticalScrollBar()
        old_value = scrollbar.value()
        at_bottom = old_value >= scrollbar.maximum() - 20
        self._text_area.setHtml(html)
        if at_bottom or self._auto_scroll_enabled:
            cursor = self._text_area.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self._text_area.setTextCursor(cursor)
        else:
            scrollbar.setValue(old_value)

    def _on_scroll(self) -> None:
        scrollbar = self._text_area.verticalScrollBar()
        self._auto_scroll_enabled = scrollbar.value() >= scrollbar.maximum() - 20

    def _clear_all(self) -> None:
        self._entries.clear()
        self._last_html = None
        self._text_area.clear()
        self._reset_storage()
        if (
            self._global_reset_checkbox.isChecked()
            and self._global_reset_handler is not None
        ):
            console.print(
                "[info][SubtitleOverlay] Triggering global reset via /global/reset...[/info]"
            )
            try:
                success = self._global_reset_handler.reset()
                if success:
                    console.print(
                        "[green][SubtitleOverlay] Global reset successful[/green]"
                    )
                else:
                    console.print(
                        "[yellow][SubtitleOverlay] Global reset returned failure[/yellow]"
                    )
            except Exception as exc:
                console.print(
                    f"[red][SubtitleOverlay] Global reset raised exception: {exc}[/red]"
                )

    def _reset_storage(self) -> None:
        if self._on_clear is not None:
            try:
                self._on_clear()
            except Exception as exc:
                console.print(
                    f"[red][ERROR][/] [SubtitleOverlay] on_clear callback failed: {exc}",
                    style="bold",
                )
        for path in self._extra_clear_paths:
            try:
                path.unlink(missing_ok=True)
            except Exception as exc:
                console.print(
                    f"[red][ERROR][/] [SubtitleOverlay] Failed to delete {path}: {exc}",
                    style="bold",
                )

    def _toggle_japanese(self) -> None:
        self._hide_japanese = self._hide_ja_btn.isChecked()
        self._last_html = None
        self._refresh_display()

    def _handle_anchor_click(self, url) -> None:
        url_str = url.toString()
        if url_str.startswith("copy:"):
            self._action_copy(int(url_str.split(":")[1]) - 1)
        elif url_str.startswith("open:"):
            self._action_open(int(url_str.split(":")[1]) - 1)
        elif url_str.startswith("play:"):
            self._action_play(int(url_str.split(":")[1]) - 1)

    def on_segment_start(self, event: SpeechSegmentStartEvent) -> None:
        pass

    def on_segment_end(self, event: SpeechSegmentEndEvent) -> None:
        pass

    def closeEvent(self, event) -> None:
        self._poll_timer.stop()
        self._sound_effect.stop()
        event.accept()

    def update_queue_status(
        self, status: str, pending: int, status_color: str = "#8b949e"
    ) -> None:
        """
        Thread-safe update of the queue status display.
        Can be called from any thread.
        """
        self.queue_status_updated.emit(status, pending, status_color)

    def _update_queue_display(
        self, status: str = None, pending: int = None, status_color: str = None
    ) -> None:
        """Update the queue status bar widgets (called on main thread via signal)."""
        if status is not None:
            self._queue_status_text = status
        if pending is not None:
            self._queue_pending_count = pending
        if status_color is not None:
            self._queue_status_color = status_color
        self._queue_status_display.setText(self._queue_status_text)
        self._queue_status_display.setStyleSheet(
            f"color: {self._queue_status_color}; font-size: 10px; font-family: monospace;"
        )
        if self._queue_pending_count > 0:
            self._queue_pending_label.setText(f"📋 {self._queue_pending_count} pending")
            self._queue_pending_label.show()
        else:
            self._queue_pending_label.hide()

    def set_retry_progress(self, visible: bool = True) -> None:
        """Show or hide the retry progress indicator."""
        if visible:
            self._retry_progress.setRange(0, 0)
            self._retry_progress.show()
        else:
            self._retry_progress.hide()
            self._retry_progress.setRange(0, 100)

    def set_retry_count(self, attempt: int) -> None:
        """Show a determinate retry count on the progress bar."""
        self._retry_progress.setRange(0, 100)
        self._retry_progress.setValue(min(attempt * 10, 100))
        self._retry_progress.show()

    def on_queue_status(
        self, status: str, pending: int, status_color: str, info: dict = None
    ) -> None:
        """QueueObserver implementation — thread-safe via Qt signal."""
        info = info or {}
        segment_num = info.get("segment_num")
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        status_entry = {
            "timestamp": timestamp,
            "status": status,
            "pending": pending,
            "color": status_color,
            "info": info,
        }
        if segment_num is not None:
            if segment_num not in self._segment_statuses:
                self._segment_statuses[segment_num] = {
                    "segment_num": segment_num,
                    "first_seen": timestamp,
                    "last_updated": timestamp,
                    "statuses": [],
                    "current_status": status,
                    "current_color": status_color,
                    "duration": info.get("duration"),
                    "start_sec": info.get("start_sec"),
                    "retry_count": 0,
                    "final_status": None,
                }
                self._segment_order.append(segment_num)
                while len(self._segment_order) > self._max_log_segments:
                    oldest_seg = self._segment_order.pop(0)
                    del self._segment_statuses[oldest_seg]
            seg_info = self._segment_statuses[segment_num]
            seg_info["last_updated"] = timestamp
            seg_info["current_status"] = status
            seg_info["current_color"] = status_color
            seg_info["statuses"].append(status_entry)
            if "retry_attempt" in info:
                seg_info["retry_count"] = info["retry_attempt"]
            status_type = info.get("status")
            if status_type == "success":
                seg_info["final_status"] = "success"
            elif status_type == "error":
                seg_info["final_status"] = "error"
            elif "✅" in status or "✓" in status or "succeeded" in status.lower():
                seg_info["final_status"] = "success"
            elif (
                "❌" in status
                or "failed" in status.lower()
                or "error" in status.lower()
            ):
                seg_info["final_status"] = "error"
        self.update_queue_status(status, pending, status_color)

    def on_retry_status(
        self, segment_num: int, attempt: int, delay: float, extra_info: dict = None
    ) -> None:
        """QueueObserver implementation for retry events."""
        status = (
            f"🔄 Retrying seg #{segment_num} (attempt {attempt}, wait {delay:.1f}s)"
        )
        info = {
            "segment_num": segment_num,
            "retry_attempt": attempt,
            "retry_delay": delay,
            **(extra_info or {}),
        }
        self.on_queue_status(status, self._queue_pending_count, "#f0883e", info)
