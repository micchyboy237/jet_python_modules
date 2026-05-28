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
- Auto-scroll that follows new entries unless the user has scrolled up
- QSoundEffect for in-app WAV playback
- VAD badge: FireRed only (hard-coded; no registry)
- Metadata row: gap, duration, balanced VAD score, speech %, speech duration,
  transcription %, coverage label
- VADScorer row: composite score and quality label with colour coding
- Implements SpeechSegmentHandler (on_segment_end is a no-op;
  subtitle data arrives via SubtitleResponseNotifier Qt signal)
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
from pathlib import Path
from typing import Optional, Sequence

from jet.audio.speech_handlers.api_types import SubtitleNotification
from jet.audio.speech_handlers.base import SpeechSegmentHandler
from jet.audio.speech_handlers.speech_events import (
    SpeechSegmentEndEvent,
    SpeechSegmentStartEvent,
)
from jet.audio.speech_handlers.subtitle_observer import SubtitleResponseNotifier
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QTextCursor
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QTextBrowser,
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
    """
    Map a balanced VAD score [0.0–1.0] → display colour.
    Uses thresholds from score_vad_segments with balanced method:
      ≥ 0.85 → Excellent (green)
      ≥ 0.70 → Good (light green)
      ≥ 0.55 → Marginal (amber)
      ≥ 0.40 → Poor (orange)
      < 0.40 → Invalid (red)
    """
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
    """
    Return a human-readable rating for a balanced VAD score.
    """
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
    """Map speech-frame percentage [0–100] → display colour."""
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
    """
    Map a VADScorer composite_score [0.0–1.0] → display colour.
    Mirrors the QUALITY_BANDS thresholds from vad_scorer.py:
      > 0.80 → Very good  (green)
      > 0.60 → Good       (light green)
      > 0.40 → Fair       (amber)
      > 0.20 → Bad        (orange)
      ≤ 0.20 → Very bad   (red)
    """
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
    """
    Map a VADScorer quality_label string → display colour.
    Provides a fallback for cases where composite_score is unavailable
    but quality_label is already known.
    """
    _MAP = {
        "Very good": "#3fb950",
        "Good": "#56d364",
        "Fair": "#e3b341",
        "Bad": "#fb923c",
        "Very bad": "#f85149",
    }
    return _MAP.get(label or "", "#8b949e")


def _speaker_confidence_color(confidence: Optional[float]) -> str:
    """Map speaker confidence [0.0–1.0] → display colour."""
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

    # Read vad_score directly from notification (computed upstream)
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
        f'<b style="font-size:10px;">{index}</b>'
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


class SubtitleOverlay(QMainWindow, SpeechSegmentHandler, metaclass=_QtABCMeta):
    """
    Always-on-top subtitle preview window.
    Subtitle data arrives via on_subtitle_received (Qt slot, main thread),
    connected to a SubtitleResponseNotifier.  on_segment_end is a no-op.
    Typical wiring via factory
    --------------------------
        overlay = SubtitleOverlay.create_and_connect(sender)
        overlay.show()
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

    def __init__(
        self,
        show_ja_text: bool = False,
        extra_clear_paths: Sequence[Path] = (),
        on_clear: Optional[callable] = None,
        parent: QWidget | None = None,
    ) -> None:
        QMainWindow.__init__(self, parent)
        self._entries: list[dict] = []
        self._hide_japanese: bool = not show_ja_text
        self._last_html: Optional[str] = None
        self._auto_scroll_enabled: bool = True
        self._extra_clear_paths: tuple[Path, ...] = tuple(extra_clear_paths)
        self._on_clear: Optional[callable] = on_clear
        self._setup_window()
        self._setup_ui()
        self._setup_sound()
        self._setup_poll_timer()

    def _setup_window(self) -> None:
        self.setWindowTitle(self._WINDOW_TITLE)
        self.resize(OVERLAY_WIDTH, OVERLAY_HEIGHT)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            self.move(geom.right() - self.width() - 20, 20)

    def _setup_ui(self) -> None:
        from jet.audio.speech_handlers.language_cache import (
            LanguageStore,
        )

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        top_bar = QHBoxLayout()

        self._clear_btn = QPushButton("🗑")
        self._clear_btn.setToolTip("Clear all entries")
        self._clear_btn.clicked.connect(self._clear_all)

        self._hide_ja_btn = QPushButton("🇯🇵")
        self._hide_ja_btn.setToolTip("Toggle Japanese text")
        self._hide_ja_btn.setCheckable(True)
        self._hide_ja_btn.setChecked(self._hide_japanese)
        self._hide_ja_btn.clicked.connect(self._toggle_japanese)

        # ── Language selector ──
        self._language_store = LanguageStore()
        self._lang_btn = QPushButton()
        self._lang_btn.setToolTip("Select transcription language")
        self._lang_btn.clicked.connect(self._cycle_language)
        self._refresh_language_button()

        top_bar.addWidget(self._clear_btn)
        top_bar.addWidget(self._hide_ja_btn)
        top_bar.addWidget(self._lang_btn)
        top_bar.addStretch()

        layout.addLayout(top_bar)

        self._text_area = QTextBrowser()
        self._text_area.setReadOnly(True)
        self._text_area.setAcceptRichText(True)
        self._text_area.setStyleSheet(self._TEXT_AREA_STYLE)
        self._text_area.setOpenExternalLinks(False)
        self._text_area.setOpenLinks(False)
        self._text_area.anchorClicked.connect(self._handle_anchor_click)
        self._text_area.verticalScrollBar().valueChanged.connect(self._on_scroll)
        layout.addWidget(self._text_area)

        self.setCentralWidget(central)

    def _refresh_language_button(self) -> None:
        """Update the language button text to reflect current selection."""
        lang = self._language_store.language
        label_map = {"auto": "🌐 Auto", "en": "🇺🇸 EN", "ja": "🇯🇵 JA"}
        self._lang_btn.setText(label_map.get(lang, f"🌐 {lang}"))

    def _cycle_language(self) -> None:
        """Cycle to the next language and update UI + sender."""
        new_lang = self._language_store.cycle_language()
        self._refresh_language_button()
        # If we have a reference to the sender, update its language too
        if hasattr(self, "_sender") and self._sender is not None:
            self._sender.set_language(new_lang)

    @property
    def language(self) -> str:
        """Expose current language so external code can read it."""
        return self._language_store.language

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
        """
        Qt slot — always runs on the Qt main thread (via queued signal).
        Maps the SubtitleNotification dict to an internal entry dict and
        appends it to the log.  Display is refreshed by the poll timer.

        VAD score is now computed upstream in WebsocketSubtitleSender and
        passed via notification["vad_score"].  This method is a pure display
        mapper with no scoring logic.
        """
        ja = notification.get("ja_text", "").strip()
        en = notification.get("en_text", "").strip()
        if not ja and not en:
            return

        # Read pre-computed vad_score directly from notification
        vad_score: Optional[float] = notification.get("vad_score")

        speaker_label = notification.get("speaker_label", "")
        speaker_confidence = notification.get("speaker_confidence")
        speaker_match_type = notification.get("speaker_match_type", "")
        diarization = notification.get("diarization", {})
        self._entries.append(
            {
                "ja": ja,
                "en": en,
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
        """Called by poll timer. Rebuilds HTML only when entries have changed."""
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

    def _reset_storage(self) -> None:
        """
        Invoke the on_clear callback (e.g. SegmentStore.reset) then delete
        any extra file paths (all_segments.json, subtitles.srt, …).
        """
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
        """No-op. Overlay is driven by WS responses, not raw segment boundaries."""

    def on_segment_end(self, event: SpeechSegmentEndEvent) -> None:
        """No-op. Subtitle text arrives via on_subtitle_received (Qt signal)."""

    def closeEvent(self, event) -> None:
        self._poll_timer.stop()
        self._sound_effect.stop()
        event.accept()

    @classmethod
    def create_and_connect(
        cls,
        sender,
        show_ja_text: bool = False,
        extra_clear_paths: Sequence[Path] = (),
        on_clear: Optional[callable] = None,
    ) -> "SubtitleOverlay":
        overlay = cls(
            show_ja_text=show_ja_text,
            extra_clear_paths=extra_clear_paths,
            on_clear=on_clear,
        )
        overlay.set_sender_reference(sender)
        notifier = SubtitleResponseNotifier(parent=overlay)
        notifier.subtitle_received.connect(overlay.on_subtitle_received)
        sender.add_observer(notifier)
        return overlay
