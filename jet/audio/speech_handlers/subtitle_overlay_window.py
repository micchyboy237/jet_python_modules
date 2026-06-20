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

import subprocess
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
from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from rich.console import Console

console = Console()

OVERLAY_WIDTH = 450
OVERLAY_HEIGHT = 600


class _QtABCMeta(type(QMainWindow), ABCMeta):
    pass


def _balanced_vad_score_color(score: Optional[float]) -> str:
    if score is None:
        return "#8b949e"
    if score >= 0.85:
        return "#3fb950"
    if score >= 0.70:
        return "#56d364"
    if score >= 0.55:
        return "#e3b341"
    if score >= 0.40:
        return "#fb923c"
    return "#f85149"


def _balanced_vad_score_rating(score: Optional[float]) -> str:
    if score is None:
        return "N/A"
    if score >= 0.85:
        return "Excellent"
    if score >= 0.70:
        return "Good"
    if score >= 0.55:
        return "Marginal"
    if score >= 0.40:
        return "Poor"
    return "Invalid"


def _speech_pctg_color(pctg: Optional[float]) -> str:
    if pctg is None:
        return "#8b949e"
    if pctg < 30:
        return "#f85149"
    if pctg < 50:
        return "#fb923c"
    if pctg < 70:
        return "#e3b341"
    return "#3fb950"


def _composite_score_color(score: Optional[float]) -> str:
    if score is None:
        return "#8b949e"
    if score > 0.80:
        return "#3fb950"
    if score > 0.60:
        return "#56d364"
    if score > 0.40:
        return "#e3b341"
    if score > 0.20:
        return "#fb923c"
    return "#f85149"


def _quality_label_color(label: Optional[str]) -> str:
    _MAP = {
        "Very good": "#3fb950",
        "Good": "#56d364",
        "Fair": "#e3b341",
        "Bad": "#fb923c",
        "Very bad": "#f85149",
    }
    return _MAP.get(label or "", "#8b949e")


def _speaker_confidence_color(confidence: Optional[float]) -> str:
    if confidence is None:
        return "#8b949e"
    if confidence > 0.70:
        return "#3fb950"
    if confidence > 0.50:
        return "#e3b341"
    if confidence > 0.30:
        return "#fb923c"
    return "#f85149"


_VAD_BADGE_HTML = (
    '<span style="'
    "background:#3b1f6b; color:#c084fc; font-family:monospace; "
    "font-size:9px; font-weight:bold; padding:1px 5px; "
    'border-radius:3px; letter-spacing:0.5px;">FRD</span>'
)


def _format_entry(
    index: int,
    entry: dict,
    prev_entry: Optional[dict],
    hide_japanese: bool,
    expanded: bool = False,
    is_playing: bool = False,
) -> str:
    segment_number = entry["segment_number"]
    ja = entry.get("ja", "").strip()
    en = entry.get("en", "").strip()
    text_display = en if hide_japanese else f"{ja}\n{en}".strip()
    start: float = entry.get("start", 0.0)
    end: float = entry.get("end", 0.0)
    end_reason = entry.get("end_reason") or "true_silence"
    segment_dir: Optional[Path] = (
        Path(entry["segment_dir"]) if entry.get("segment_dir") else None
    )
    gap: Optional[float] = None
    if prev_entry is not None:
        prev_end_time_utc = prev_entry.get("end_time_utc")
        prev_end_sec = prev_entry.get("end")
        current_start_time_utc = entry.get("start_time_utc")
        current_start_sec = entry.get("start")
        if current_start_time_utc and prev_end_time_utc:
            from datetime import datetime

            try:
                start_dt = datetime.fromisoformat(current_start_time_utc)
                end_dt = datetime.fromisoformat(prev_end_time_utc)
                gap = (start_dt - end_dt).total_seconds()
            except (ValueError, TypeError):
                pass
        if gap is None and current_start_sec is not None and prev_end_sec is not None:
            gap = current_start_sec - prev_end_sec
    gap_str = f"{gap:.2f}s" if gap is not None else "—"
    duration = end - start
    vad_score: Optional[float] = entry.get("vad_score")
    vad_score_str = f"{vad_score:.3f}" if isinstance(vad_score, float) else "N/A"
    vad_score_color = _balanced_vad_score_color(vad_score)
    vad_rating = _balanced_vad_score_rating(vad_score)
    speech_pctg: Optional[float] = entry.get("speech_frames_pctg")
    speech_pctg_str = (
        f"{speech_pctg:.1f}%" if isinstance(speech_pctg, (int, float)) else "N/A"
    )
    speech_pctg_color = _speech_pctg_color(speech_pctg)
    transcribed_pctg = entry.get("transcribed_duration_pctg")
    trans_pctg_str = (
        f"{float(transcribed_pctg):.1f}%"
        if isinstance(transcribed_pctg, (int, float))
        else "N/A"
    )
    trans_pctg_color = _speech_pctg_color(
        float(transcribed_pctg) if isinstance(transcribed_pctg, (int, float)) else None
    )
    composite_score: Optional[float] = entry.get("vad_composite_score")
    composite_str = (
        f"{composite_score:.3f}" if isinstance(composite_score, (int, float)) else "N/A"
    )
    composite_color = _composite_score_color(
        float(composite_score) if isinstance(composite_score, (int, float)) else None
    )
    quality_label: Optional[str] = entry.get("vad_quality_label")
    quality_str = quality_label or "N/A"
    quality_color = _quality_label_color(quality_label)
    speaker_label = entry.get("speaker_label", "")
    speaker_confidence: Optional[float] = entry.get("speaker_confidence")
    speaker_conf_str = (
        f" ({speaker_confidence:.2f})" if isinstance(speaker_confidence, float) else ""
    )
    speaker_conf_color = _speaker_confidence_color(speaker_confidence)
    speaker_match_type = entry.get("speaker_match_type", "")
    copy_link = (
        f'<a href="copy:{index}" style="color:#58a6ff; text-decoration:none;">📋</a>'
    )
    open_link = (
        f'<a href="open:{index}" style="color:#58a6ff; text-decoration:none;">📂</a>'
        if segment_dir
        else ""
    )
    play_icon = "⏸" if is_playing else "▶"
    play_link = (
        f'<a href="play:{index}" style="color:#ff7b72; text-decoration:none;'
        f' font-size:13px;">{play_icon}</a>'
        if segment_dir and (segment_dir / "sound.wav").exists()
        else ""
    )
    speaker_badge = ""
    if speaker_label:
        speaker_badge = (
            f' <span style="'
            f"background:#1f3b5c; color:#79c0ff; font-family:monospace; "
            f"font-size:9px; font-weight:bold; padding:1px 5px; "
            f'border-radius:3px;">{speaker_label}'
            f'<span style="color:{speaker_conf_color};">{speaker_conf_str}</span>'
            f"</span>"
        )
    header_html = (
        f'<b style="font-size:10px;">{segment_number}</b>'
        f"{speaker_badge} "
        f'<span style="font-size:9px; color:#8b949e;">'
        f"({duration:.2f}s)"
        f" • gap: {gap_str}"
        f' • <span style="color:#d2a8ff;">{end_reason}</span>'
        f' • VAD: <span style="color:{vad_score_color}; font-weight:bold;">{vad_score_str}</span>'
        f' <span style="color:{vad_score_color}; font-size:8px;">({vad_rating})</span>'
        f"</span>"
        f" {copy_link} {open_link} {play_link}"
    )
    text_html = (
        f'<div style="margin-top:5px; margin-bottom:2px;">'
        f'<span style="'
        f"font-size:13px; "
        f"color:#e6edf3; "
        f"font-family:'Segoe UI', 'SF Pro Text', Arial, sans-serif; "
        f"line-height:1.5; "
        f"letter-spacing:0.5px;"
        f'">{text_display.replace(chr(10), "<br/>")}</span>'
        f"</div>"
    )
    return (
        f'<div style="margin-bottom:4px;">'
        f"{header_html}"
        f"{text_html}"
        f"</div>"
        f'<hr style="border:none; border-top:1px solid #30363d; margin:4px 0;">'
    )


class SubtitleOverlay(
    QMainWindow, SpeechSegmentHandler, QueueObserver, metaclass=_QtABCMeta
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

    # Queue status signal (thread-safe)
    queue_status_updated = pyqtSignal(
        str, int, str
    )  # (status_text, pending_count, status_color)

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

        # Register overlay as queue observer
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
        # NEW: Segment-based status tracking
        self._segment_statuses: dict[int, dict] = {}  # seg_num -> segment status info
        self._segment_order: list[int] = []  # ordered list of segment numbers
        self._max_log_segments: int = 25  # keep last N segments
        self._log_viewer: Optional[QDialog] = None

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

        # ── Top bar with action buttons ──
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

        # ── Queue Status Bar with Log History Button ──
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

        # Log history button
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

        # Progress bar for retry indication
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
        # ── End Queue Status Bar ──

        # ── Main text area ──
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
        html_parts: list[str] = []
        prev_entry: Optional[dict] = None
        for i, entry in enumerate(self._entries, 1):
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

    def _action_copy(self, idx: int) -> None:
        if not (0 <= idx < len(self._entries)):
            return
        e = self._entries[idx]
        text = f"{e.get('ja', '')}\n{e.get('en', '')}".strip()
        QApplication.clipboard().setText(text)
        self.setWindowTitle("Copied ✓")
        QTimer.singleShot(800, lambda: self.setWindowTitle(self._WINDOW_TITLE))

    def _action_open(self, idx: int) -> None:
        if not (0 <= idx < len(self._entries)):
            return
        segment_dir = self._entries[idx].get("segment_dir")
        if segment_dir:
            try:
                subprocess.Popen(["open", str(segment_dir)])
            except Exception:
                pass

    def _action_play(self, idx: int) -> None:
        if not (0 <= idx < len(self._entries)):
            return
        segment_dir = self._entries[idx].get("segment_dir")
        if not segment_dir:
            return
        wav_path = Path(segment_dir) / "sound.wav"
        if not wav_path.exists():
            return
        url = QUrl.fromLocalFile(str(wav_path))
        self._sound_effect.setSource(url)
        self._sound_effect.setVolume(1.0)
        self._sound_effect.play()

    def on_segment_start(self, event: SpeechSegmentStartEvent) -> None:
        pass

    def on_segment_end(self, event: SpeechSegmentEndEvent) -> None:
        pass

    def closeEvent(self, event) -> None:
        self._poll_timer.stop()
        self._sound_effect.stop()
        event.accept()

    # --- Queue Status Management ---

    def update_queue_status(
        self, status: str, pending: int, status_color: str = "#8b949e"
    ) -> None:
        """
        Thread-safe update of the queue status display.
        Can be called from any thread.
        """
        self.queue_status_updated.emit(status, pending, status_color)
        # console.print(
        #     f"[debug][QueueStatus] {status} | Pending: {pending} | Color: {status_color}[/debug]"
        # )

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
            self._retry_progress.setRange(0, 0)  # Indeterminate
            self._retry_progress.show()
        else:
            self._retry_progress.hide()
            self._retry_progress.setRange(0, 100)

    def set_retry_count(self, attempt: int) -> None:
        """Show a determinate retry count on the progress bar."""
        self._retry_progress.setRange(0, 100)
        self._retry_progress.setValue(min(attempt * 10, 100))
        self._retry_progress.show()

    # --- QueueObserver Implementation ---

    def on_queue_status(
        self, status: str, pending: int, status_color: str, info: dict = None
    ) -> None:
        """QueueObserver implementation — thread-safe via Qt signal."""
        info = info or {}
        segment_num = info.get("segment_num")
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]

        # Create a status entry for the event
        status_entry = {
            "timestamp": timestamp,
            "status": status,
            "pending": pending,
            "color": status_color,
            "info": info,
        }

        # Group by segment if segment_num is provided
        if segment_num is not None:
            if segment_num not in self._segment_statuses:
                # Initialize new segment entry
                self._segment_statuses[segment_num] = {
                    "segment_num": segment_num,
                    "first_seen": timestamp,
                    "last_updated": timestamp,
                    "statuses": [],  # list of all status events for this segment
                    "current_status": status,
                    "current_color": status_color,
                    "duration": info.get("duration"),
                    "start_sec": info.get("start_sec"),
                    "retry_count": 0,
                    "final_status": None,  # will be set when segment completes
                }
                self._segment_order.append(segment_num)

                # Trim to max segments
                while len(self._segment_order) > self._max_log_segments:
                    oldest_seg = self._segment_order.pop(0)
                    del self._segment_statuses[oldest_seg]

            # Update segment info
            seg_info = self._segment_statuses[segment_num]
            seg_info["last_updated"] = timestamp
            seg_info["current_status"] = status
            seg_info["current_color"] = status_color
            seg_info["statuses"].append(status_entry)

            # Track retries
            if "retry_attempt" in info:
                seg_info["retry_count"] = info["retry_attempt"]

            # Detect completion from info dict first (most reliable)
            status_type = info.get("status")
            if status_type == "success":
                seg_info["final_status"] = "success"
            elif status_type == "error":
                seg_info["final_status"] = "error"
            # Fallback: detect from status text
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

        # Create info dict for queue status
        info = {
            "segment_num": segment_num,
            "retry_attempt": attempt,
            "retry_delay": delay,
            **(extra_info or {}),
        }

        # Update segment status through the main path
        self.on_queue_status(status, self._queue_pending_count, "#f0883e", info)

    def _show_log_history(self) -> None:
        """Show a dialog with per-segment status history that updates live."""
        from PyQt6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout

        if self._log_viewer is not None:
            self._log_viewer.close()
            self._log_viewer = None

        dialog = QDialog(self)
        dialog.setWindowTitle("📜 Queue Status History ● LIVE")
        dialog.resize(650, 550)
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header
        header_layout = QHBoxLayout()
        live_indicator = QLabel("● LIVE")
        live_indicator.setStyleSheet(
            "color: #3fb950; font-size: 9px; font-weight: bold; "
            "font-family: 'Consolas', monospace;"
        )

        segment_count_label = QLabel()
        segment_count_label.setStyleSheet(
            "color: #6e7681; font-size: 9px; font-family: 'Consolas', monospace;"
        )

        header_layout.addWidget(live_indicator)
        header_layout.addStretch()
        header_layout.addWidget(segment_count_label)
        layout.addLayout(header_layout)

        # Text area for history
        text_area = QTextEdit()
        text_area.setReadOnly(True)
        text_area.setStyleSheet("""
            QTextEdit {
                background-color: #0d1117;
                color: #c9d1d9;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                border: 1px solid #30363d;
            }
            QScrollBar:vertical {
                background-color: #161b22;
                width: 8px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background-color: #30363d;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #484f58;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        # Bottom bar
        bottom_bar = QHBoxLayout()

        auto_scroll_cb = QCheckBox("Auto-scroll")
        auto_scroll_cb.setChecked(True)
        auto_scroll_cb.setStyleSheet("""
            QCheckBox {
                color: #8b949e;
                font-size: 10px;
                spacing: 4px;
            }
            QCheckBox::indicator {
                width: 12px;
                height: 12px;
                border: 1px solid #30363d;
                border-radius: 2px;
                background-color: #21262d;
            }
            QCheckBox::indicator:checked {
                background-color: #238636;
                border-color: #2ea043;
            }
        """)

        copy_btn = QPushButton("📋 Copy Logs")
        copy_btn.setToolTip("Copy all segment history to clipboard as formatted text")
        copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 3px;
                color: #58a6ff;
                padding: 4px 10px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #1f3b5c;
                border-color: #58a6ff;
            }
        """)

        clear_btn = QPushButton("Clear History")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 3px;
                color: #f85149;
                padding: 4px 10px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #3d1f1f;
                border-color: #f85149;
            }
        """)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 3px;
                color: #c9d1d9;
                padding: 4px 16px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #30363d;
            }
        """)

        bottom_bar.addWidget(auto_scroll_cb)
        bottom_bar.addWidget(copy_btn)
        bottom_bar.addWidget(clear_btn)
        bottom_bar.addStretch()
        bottom_bar.addWidget(close_btn)
        layout.addLayout(bottom_bar)

        def _get_status_icon(
            status_text: str, final_status: Optional[str], info: dict = None
        ) -> str:
            """Determine the appropriate icon based on segment status."""
            # Check info dict first for explicit status
            if info and info.get("status") == "success":
                return "✅"
            if info and info.get("status") == "error":
                return "❌"

            # Check final_status
            if final_status == "success":
                return "✅"
            if final_status == "error":
                return "❌"

            # Check status text patterns
            if "⏳" in status_text or "sending" in status_text.lower():
                return "⏳"
            elif "🔄" in status_text or "retry" in status_text.lower():
                return "🔄"
            elif "✅" in status_text or "succeeded" in status_text.lower():
                return "✅"
            elif "❌" in status_text or "failed" in status_text.lower():
                return "❌"
            elif "📋" in status_text or "queued" in status_text.lower():
                return "📋"
            else:
                return "📡"

        def _get_status_color(final_status: Optional[str], current_color: str) -> str:
            """Get color based on final status."""
            if final_status == "success":
                return "#3fb950"
            elif final_status == "error":
                return "#f85149"
            return current_color

        def _generate_plain_text_log() -> str:
            """Generate a plain text version of the segment history for clipboard."""
            lines = []
            lines.append("=" * 70)
            lines.append("QUEUE STATUS HISTORY - SEGMENT LOG")
            lines.append(
                f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )
            lines.append(f"Total Segments: {len(self._segment_order)}")
            lines.append("=" * 70)
            lines.append("")

            for seg_num in self._segment_order:
                if seg_num not in self._segment_statuses:
                    continue

                seg_info = self._segment_statuses[seg_num]
                duration = seg_info.get("duration")
                start_sec = seg_info.get("start_sec")
                retry_count = seg_info.get("retry_count", 0)
                final_status = seg_info.get("final_status", "pending")
                first_seen = seg_info["first_seen"]
                last_updated = seg_info["last_updated"]
                statuses = seg_info["statuses"]

                # Segment header
                status_str = final_status.upper() if final_status else "PENDING"
                lines.append(f"[{status_str}] Segment #{seg_num}")
                if duration:
                    lines.append(f"  Duration: {duration:.2f}s")
                if start_sec is not None:
                    lines.append(f"  Start: {start_sec:.2f}s")
                if retry_count > 0:
                    lines.append(f"  Retries: {retry_count}")
                lines.append(f"  Time: {first_seen} → {last_updated}")
                lines.append("")

                # Status timeline
                lines.append("  Status Timeline:")
                for entry in statuses:
                    icon = _get_status_icon(entry["status"], None, entry.get("info"))
                    lines.append(f"    {entry['timestamp']} {icon} {entry['status']}")

                    # Add extra info if available
                    info = entry.get("info", {})
                    if info.get("error"):
                        lines.append(f"      Error: {info['error']}")
                    if info.get("retry_attempt"):
                        lines.append(
                            f"      Attempt: {info['retry_attempt']}, Delay: {info.get('retry_delay', 0):.1f}s"
                        )

                lines.append("")
                lines.append("-" * 70)
                lines.append("")

            return "\n".join(lines)

        def render_log():
            """Build and display per-segment status history."""
            # Get segments in chronological order (most recent last)
            segments = []
            for seg_num in self._segment_order:
                if seg_num in self._segment_statuses:
                    segments.append(self._segment_statuses[seg_num])

            segment_count_label.setText(
                f"{len(segments)} segments (last {self._max_log_segments})"
            )

            if not segments:
                text_area.setHtml(
                    '<div style="color: #6e7681; padding: 30px; text-align: center;">'
                    "No segments processed yet<br>"
                    '<span style="font-size: 9px;">Start recording to see segment status</span>'
                    "</div>"
                )
                return

            html_parts = ['<div style="font-family: monospace;">']

            for seg_info in segments:
                seg_num = seg_info["segment_num"]
                duration = seg_info.get("duration")
                start_sec = seg_info.get("start_sec")
                retry_count = seg_info.get("retry_count", 0)
                final_status = seg_info.get("final_status")
                current_status = seg_info["current_status"]
                current_color = seg_info["current_color"]
                first_seen = seg_info["first_seen"]
                last_updated = seg_info["last_updated"]
                statuses = seg_info["statuses"]

                # Get the latest status entry for more accurate icon detection
                latest_info = statuses[-1].get("info", {}) if statuses else {}
                icon = _get_status_icon(current_status, final_status, latest_info)
                color = _get_status_color(final_status, current_color)

                # Build segment header
                duration_str = f" {duration:.1f}s" if duration else ""
                start_str = f" @{start_sec:.1f}s" if start_sec is not None else ""
                retry_str = f" [Retries: {retry_count}]" if retry_count > 0 else ""

                # Status badge
                status_badge = ""
                if final_status == "success":
                    status_badge = (
                        '<span style="background:#1b3d1b; color:#3fb950; '
                        'font-size:8px; padding:1px 4px; border-radius:2px; margin-left:6px;">'
                        "SUCCESS</span>"
                    )
                elif final_status == "error":
                    status_badge = (
                        '<span style="background:#3d1f1f; color:#f85149; '
                        'font-size:8px; padding:1px 4px; border-radius:2px; margin-left:6px;">'
                        "ERROR</span>"
                    )

                header_html = (
                    f'<div style="'
                    f"background-color: #161b22; "
                    f"border-left: 3px solid {color}; "
                    f"padding: 6px 8px; "
                    f"margin-bottom: 4px; "
                    f"border-radius: 4px;"
                    f'">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                    f"<span>"
                    f'<span style="color: {color}; font-weight: bold; font-size: 12px;">'
                    f"{icon} Segment #{seg_num}{duration_str}{start_str}"
                    f"</span>"
                    f"{status_badge}"
                    f"</span>"
                    f'<span style="color: #6e7681; font-size: 9px;">'
                    f"{first_seen} → {last_updated}"
                    f"</span>"
                    f"</div>"
                )

                if retry_str:
                    header_html += (
                        f'<div style="margin-top: 3px;">'
                        f'<span style="color: #f0883e; font-size: 9px;">{retry_str}</span>'
                        f"</div>"
                    )

                # Add status timeline (collapsed by default, expandable)
                if statuses:
                    # Show all statuses for completed segments, last 3 for active ones
                    if final_status:
                        display_statuses = statuses  # Show all for completed
                    else:
                        display_statuses = (
                            statuses[-3:] if len(statuses) > 3 else statuses
                        )

                    header_html += (
                        '<div style="margin-top: 4px; font-size: 9px; color: #8b949e;">'
                    )

                    for status_entry in display_statuses:
                        entry_info = status_entry.get("info", {})
                        status_icon = _get_status_icon(
                            status_entry["status"], None, entry_info
                        )

                        # Add error details if present
                        error_detail = ""
                        if entry_info.get("error"):
                            error_detail = (
                                f' <span style="color: #f85149;">'
                                f"({str(entry_info['error'])[:50]})"
                                f"</span>"
                            )

                        header_html += (
                            f'<div style="margin-left: 8px; opacity: 0.85;">'
                            f'<span style="color: #6e7681;">{status_entry["timestamp"]}</span> '
                            f"{status_icon} "
                            f'<span style="color: {status_entry["color"]};">'
                            f"{status_entry['status'][:80]}"
                            f"</span>"
                            f"{error_detail}"
                            f"</div>"
                        )

                    if not final_status and len(statuses) > 3:
                        header_html += (
                            f'<div style="margin-left: 8px; color: #6e7681; font-style: italic;">'
                            f"... and {len(statuses) - 3} more events"
                            f"</div>"
                        )

                    header_html += "</div>"

                header_html += "</div>"
                html_parts.append(header_html)

            html_parts.append("</div>")

            # Preserve scroll position
            scrollbar = text_area.verticalScrollBar()
            was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 10

            text_area.setHtml("".join(html_parts))

            if auto_scroll_cb.isChecked() and was_at_bottom:
                scrollbar.setValue(scrollbar.maximum())

        def copy_logs_to_clipboard():
            """Copy the plain text log to clipboard."""
            plain_text = _generate_plain_text_log()
            QApplication.clipboard().setText(plain_text)

            # Visual feedback - briefly change button text
            original_text = copy_btn.text()
            copy_btn.setText("✓ Copied!")
            copy_btn.setStyleSheet("""
                QPushButton {
                    background-color: #1b3d1b;
                    border: 1px solid #3fb950;
                    border-radius: 3px;
                    color: #3fb950;
                    padding: 4px 10px;
                    font-size: 10px;
                }
            """)
            QTimer.singleShot(1500, lambda: restore_copy_button(original_text))

        def restore_copy_button(original_text: str):
            """Restore copy button to original state."""
            copy_btn.setText(original_text)
            copy_btn.setStyleSheet("""
                QPushButton {
                    background-color: #21262d;
                    border: 1px solid #30363d;
                    border-radius: 3px;
                    color: #58a6ff;
                    padding: 4px 10px;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: #1f3b5c;
                    border-color: #58a6ff;
                }
            """)

        # Initial render
        render_log()
        layout.insertWidget(1, text_area)

        # Live update timer
        live_timer = QTimer(dialog)
        live_timer.timeout.connect(render_log)
        live_timer.start(500)

        # Button connections
        copy_btn.clicked.connect(copy_logs_to_clipboard)
        clear_btn.clicked.connect(lambda: self._clear_segment_history())
        close_btn.clicked.connect(dialog.close)

        def on_dialog_closed():
            live_timer.stop()
            if self._log_viewer is dialog:
                self._log_viewer = None
            console.print("[debug][LogHistory] Live segment viewer closed[/debug]")

        dialog.finished.connect(on_dialog_closed)

        self._log_viewer = dialog
        dialog.show()
        console.print(
            "[debug][LogHistory] Live segment viewer opened — updating every 500ms[/debug]"
        )

    def _clear_segment_history(self) -> None:
        """Clear the segment status history."""
        self._segment_statuses.clear()
        self._segment_order.clear()
        console.print("[debug][SubtitleOverlay] Segment status history cleared[/debug]")
