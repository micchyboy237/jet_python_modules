# live_subtitles_overlay.py
# Usage Example:
# overlay = LiveSubtitlesOverlay.create()
# overlay.add_message("Your text here")   # Exactly what you want!

import asyncio
import logging
import signal
import sys
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from pathlib import Path
from typing import NotRequired, TypedDict

from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name
from PyQt6.QtCore import (
    QDir,
    QEasingCurve,
    QObject,
    QPoint,
    QPropertyAnimation,
    Qt,
    QTimer,
    QUrl,
    pyqtSignal,
)
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from rich.logging import RichHandler

PLAY_VOLUME = 1.0
MINIMIZED_HEIGHT = 600


def _setup_logging():
    logger = logging.getLogger("LiveSubtitlesOverlay")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True, markup=True, show_path=False)
        handler.setFormatter(
            logging.Formatter("[bold magenta]Subtitle[/] → %(message)s")
        )
        logger.addHandler(handler)
    return logger


class SubtitleMessage(TypedDict):
    id: str
    translated_text: str
    start_sec: float
    end_sec: float
    duration_sec: float
    source_text: str

    segment_number: NotRequired[int]
    is_partial: NotRequired[bool]
    chunk_index: NotRequired[int]

    avg_vad_confidence: NotRequired[float]
    rms: NotRequired[float]
    rms_label: NotRequired[str]
    transcription_confidence: NotRequired[float]
    translation_confidence: NotRequired[float]
    # Removed quality labels from display (still allowed in data)
    transcription_quality: NotRequired[str]
    translation_quality: NotRequired[str]


class _Signals(QObject):
    # Signals for thread-safe operation
    _add_message = pyqtSignal(object)  # emits full SubtitleMessage dict
    _clear = pyqtSignal()
    _toggle_minimize = pyqtSignal()
    _enqueue_task = pyqtSignal(object, dict)  # (coro: Awaitable, kwargs: dict)


class LiveSubtitlesOverlay(QWidget):
    """
    Modern, thread-safe, live overlay for displaying subtitles, transcripts, or status messages.

    Example:
        overlay = LiveSubtitlesOverlay.create()
        overlay.add_message("Hello world!")        # Always safe and simple
        overlay.add_message("Another line")
        overlay.clear()
    """

    def __init__(
        self,
        parent=None,
        title: str | None = None,
        on_clear: Callable[[], None] | None = None,
        play_volume: float = PLAY_VOLUME,
        segments_dir: str | None = None,
        hide_source_text: bool = False,
    ):
        super().__init__(parent)
        self.logger = _setup_logging()
        self._on_clear_callback = on_clear
        self.signals = _Signals()
        self.history = []
        self.title = title
        self.hide_source_text = hide_source_text
        self.message_history: list[SubtitleMessage] = []
        self._task_lock = threading.Lock()  # for thread-safety with add_task

        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 190); border-radius: 12px;")
        self.setFixedSize(450, 650)

        self._drag_pos = QPoint()
        self._build_ui()
        self._connect_signals()
        self.signals._enqueue_task.connect(self._on_enqueue_task)

        # Async integration/setup
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._async_loop = None
        self._async_loop_thread = None
        self._start_async_loop()

        self._pending_tasks: list[
            tuple[Awaitable[SubtitleMessage | str], QWidget, QHBoxLayout]
        ] = []

        # Initial message
        self.logger.info("[green]Ready – use .add_message('text')[/]")
        self._process_next_task()

        self._message_by_id: dict[str, SubtitleMessage] = {}
        self._widget_by_id: dict[str, QWidget] = {}
        # Shared media player for segment playback
        self._player = QMediaPlayer()

        self._opacity_effects: dict[QWidget, QGraphicsOpacityEffect] = {}
        self._widgets_per_segment: dict[
            int, list[QWidget]
        ] = {}  # all widgets by segment

        # Track last rendered segment for UI grouping
        self._last_rendered_segment: int | None = None

        self._audio_output = QAudioOutput()
        self._audio_output.setVolume(play_volume)  # 0.0–1.0; tweak as needed
        self._player.setAudioOutput(self._audio_output)
        # Base directory where segments are saved (adjust if needed)
        self.segments_dir = str(
            segments_dir
            or Path(get_entry_file_dir())
            / "generated"
            / Path(get_entry_file_name()).stem
            / "segments"
        )

        # persistent animations so they don't get garbage-collected
        self._active_animations: list[QPropertyAnimation] = []

    def _copy_to_clipboard(self, text: str, what: str = "text") -> None:
        if not text.strip():
            return
        QApplication.clipboard().setText(text)
        # Brief non-disruptive feedback
        orig = self.status_label.text()
        self.status_label.setText(f"Copied {what}")
        QTimer.singleShot(
            1400,
            lambda: self.status_label.setText(
                orig if orig != "LIVE • MINIMIZED" else "LIVE"
            ),
        )

    def _start_async_loop(self):
        def run_loop():
            self._async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._async_loop)
            self._async_loop.run_forever()

        self._async_loop_thread = threading.Thread(
            target=run_loop, daemon=True, name="OverlayAsyncLoop"
        )
        self._async_loop_thread.start()
        time.sleep(0.05)

    def _process_next_task(self) -> None:
        if not self._pending_tasks:
            return

        coro, loading_widget, current_layout = self._pending_tasks[0]

        # Remove pending widgets from the layout
        while current_layout.count():
            item = current_layout.takeAt(0)
            if w := item.widget():
                w.deleteLater()

        spinner = QLabel()
        self._setup_spinner_animation(spinner, loading_widget)
        processing_text = QLabel("Processing")
        processing_text.setStyleSheet("color: #4da6ff;")
        processing_text.setFont(
            QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 13)
        )  # 15 → 13
        current_layout.addStretch()
        current_layout.addWidget(spinner)
        current_layout.addWidget(processing_text)
        current_layout.addStretch()

        def run_in_thread() -> tuple[SubtitleMessage | str, QWidget]:
            try:
                future = asyncio.run_coroutine_threadsafe(coro, self._async_loop)
                result = future.result()
                return result, loading_widget
            except Exception as e:
                self.logger.exception("Task failed")
                return f"[Error] {e}", loading_widget

        future = self._executor.submit(run_in_thread)
        QTimer.singleShot(100, lambda: self._check_future_done(future))

    def _check_future_done(self, future: Future) -> None:
        if future.done():
            self._on_task_finished(future)
        else:
            QTimer.singleShot(100, lambda: self._check_future_done(future))

    def _on_task_finished(self, future: Future) -> None:
        if future.cancelled():
            return

        if self._pending_tasks:
            self._pending_tasks.pop(0)

        try:
            result, loading_widget = future.result()
        except Exception as e:
            result = f"[Error] {e}"

        if not loading_widget or not loading_widget.parent():
            if self._pending_tasks:
                self._process_next_task()
            return

        timer = loading_widget.property("_spinner_timer")
        if timer and isinstance(timer, QTimer) and timer.isActive():
            timer.stop()
            timer.deleteLater()

        idx = self.content_layout.indexOf(loading_widget)
        if idx == -1:
            if self._pending_tasks:
                self._process_next_task()
            return

        self.content_layout.takeAt(idx)
        loading_widget.deleteLater()

        if isinstance(result, dict) and "translated_text" in result:
            subtitle_message: SubtitleMessage = {
                "translated_text": str(result.get("translated_text", "")),
                "start_sec": float(result.get("start_sec", 0.0)),
                "end_sec": float(result.get("end_sec", 0.0)),
                "duration_sec": float(result.get("duration_sec", 0.0)),
                "source_text": str(result.get("source_text", "")),
                **(
                    {
                        k: v
                        for k, v in {
                            "segment_number": result.get("segment_number"),
                            "avg_vad_confidence": result.get("avg_vad_confidence"),
                            "transcription_confidence": result.get(
                                "transcription_confidence"
                            ),
                        }.items()
                        if v is not None
                    }
                ),
            }
            self.message_history.append(subtitle_message)
            self.history.append(subtitle_message["translated_text"])
            self.signals._add_message.emit(subtitle_message)
        else:
            text = str(result)
            new_label = QLabel(text)
            new_label.setWordWrap(True)
            new_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            new_label.setStyleSheet("color: white; padding: 4px;")
            new_label.setFont(
                QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 15)
            )
            self.content_layout.insertWidget(idx, new_label)
            self.history.append(text)

        QTimer.singleShot(
            0,
            lambda: self.scroll.verticalScrollBar().setValue(
                self.scroll.verticalScrollBar().maximum()
            ),
        )

        if self._pending_tasks:
            self._process_next_task()

    def closeEvent(self, event):
        self._executor.shutdown(wait=False)
        if self._async_loop:
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        if self._async_loop_thread and self._async_loop_thread.is_alive():
            self._async_loop_thread.join(timeout=3.0)
        super().closeEvent(event)

    def _build_ui(self):
        # === IMPORTS LOCAL TO METHOD (to fix UnboundLocalError) ===
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import (
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QRadioButton,
            QToolButton,
            QVBoxLayout,
            QWidget,
        )

        main = QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(4)

        # Top bar: title, status, buttons
        self.control_bar = QHBoxLayout()
        self.control_bar.setContentsMargins(8, 6, 8, 6)
        self.control_bar.setSpacing(8)

        if self.title:
            self.title_label = QLabel(self.title)
            self.title_label.setStyleSheet(
                "color: #ffffff; font-size: 14px; font-weight: bold;"
            )
            self.control_bar.addWidget(self.title_label)

        self.status_label = QLabel("LIVE")
        self.status_label.setStyleSheet("""
            color: #ff4444;
            background: rgba(255, 50, 50, 0.25);
            border: 1px solid rgba(255, 100, 100, 0.4);
            border-radius: 6px;
            padding: 2px 8px;
            font-weight: bold;
            font-size: 12px;
        """)
        self.control_bar.addWidget(self.status_label)
        self.control_bar.addStretch()

        # === VAD FILTER TOGGLE BUTTON ===
        self.filter_toggle_btn = QToolButton()
        self.filter_toggle_btn.setText("⚙ VAD Filter")
        self.filter_toggle_btn.setCheckable(True)
        self.filter_toggle_btn.setFixedSize(78, 28)
        self.filter_toggle_btn.setStyleSheet("""
            QToolButton {
                background: rgba(100, 140, 255, 0.25);
                color: white;
                border-radius: 14px;
                font-weight: bold;
                font-size: 11px;
            }
            QToolButton:checked {
                background: rgba(100, 140, 255, 0.6);
            }
            QToolButton:hover {
                background: rgba(100, 140, 255, 0.5);
            }
        """)
        self.filter_toggle_btn.toggled.connect(self._toggle_filter_panel)
        self.control_bar.addWidget(self.filter_toggle_btn)

        clear_btn = QPushButton("🗑")
        clear_btn.setFixedSize(28, 28)
        clear_btn.setStyleSheet("""
            QPushButton {
                background: rgba(200, 80, 80, 0.35);
                color: #ffffff;
                border-radius: 14px;
                font-size: 15px;
            }
            QPushButton:hover { background: rgba(255, 70, 70, 0.65); }
            QPushButton:pressed { background: rgba(200, 40, 40, 0.9); }
        """)
        clear_btn.clicked.connect(self.clear)

        self.min_btn = QPushButton("−")
        self.min_btn.setFixedSize(28, 28)
        self.min_btn.setStyleSheet("""
            QPushButton {
                background: rgba(90, 140, 200, 0.35);
                color: #ffffff;
                border-radius: 14px;
                font-size: 17px;
            }
            QPushButton:hover { background: rgba(100, 160, 220, 0.6); }
            QPushButton:pressed { background: rgba(70, 120, 180, 0.85); }
        """)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(28, 28)
        close_btn.setStyleSheet("""
            QPushButton {
                background: rgba(220, 60, 60, 0.85);
                color: white;
                border-radius: 14px;
                font-size: 17px;
            }
            QPushButton:hover { background: rgba(255, 50, 50, 1.0); }
            QPushButton:pressed { background: rgba(180, 30, 30, 1.0); }
        """)
        close_btn.clicked.connect(QApplication.quit)

        self.control_bar.addWidget(clear_btn)
        self.control_bar.addWidget(self.min_btn)
        self.control_bar.addWidget(close_btn)

        # === VAD FILTER PANEL (radio buttons, full width when shown) ===
        self.filter_group = QGroupBox("VAD Confidence Filter")
        self.filter_group.setStyleSheet("""
            QGroupBox {
                font-size: 11px;
                font-weight: bold;
                color: #ffffff;
                margin-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px 0 4px;
            }
        """)

        # Inner widget + layout to stretch full width
        filter_widget = QWidget()
        filter_layout = QHBoxLayout(filter_widget)
        filter_layout.setContentsMargins(12, 12, 12, 12)
        filter_layout.setSpacing(20)

        self.vad_all = QRadioButton("All")
        self.vad_high = QRadioButton("High (≥0.7)")
        self.vad_med = QRadioButton("Medium (≥0.5)")

        for btn in (self.vad_all, self.vad_high, self.vad_med):
            btn.setStyleSheet("color: white; font-size: 11px;")
            filter_layout.addWidget(btn)

        filter_layout.addStretch()  # push buttons left

        self.vad_all.setChecked(True)  # default
        self.vad_all.toggled.connect(self._apply_filters)
        self.vad_high.toggled.connect(self._apply_filters)
        self.vad_med.toggled.connect(self._apply_filters)

        self.filter_group.setLayout(filter_layout)
        self.filter_group.setVisible(False)

        # Insert filter group directly into main layout (below control bar)
        main.addWidget(self.filter_group)

        self.control_bar_widget = QWidget()
        self.control_bar_widget.setLayout(self.control_bar)
        self.control_bar_widget.setCursor(Qt.CursorShape.OpenHandCursor)
        main.addWidget(self.control_bar_widget)

        # Content area (unchanged)
        self.content_area = QVBoxLayout()
        self.content_area.setSpacing(4)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.content_layout.setSpacing(4)
        self.content_layout.setContentsMargins(6, 6, 6, 6)
        self.content_layout.addStretch(1)
        self.scroll.setWidget(self.content)

        self.content_area.addWidget(self.scroll, stretch=1)
        main.addLayout(self.content_area, stretch=1)

        self.min_btn.clicked.connect(self.toggle_minimize)
        self._is_minimized = False
        self._original_size = None

    def _toggle_filter_panel(self, checked: bool) -> None:
        """Show/hide the VAD filter panel"""
        self.filter_group.setVisible(checked)
        if checked:
            self.filter_toggle_btn.setText("✓ VAD Filter")
        else:
            self.filter_toggle_btn.setText("⚙ VAD Filter")

    def _apply_filters(self) -> None:
        """Re-render the entire visible list based on the selected VAD filter"""
        if self.vad_all.isChecked():
            min_vad = 0.0
        elif self.vad_high.isChecked():
            min_vad = 0.7
        elif self.vad_med.isChecked():
            min_vad = 0.5
        else:
            min_vad = 0.0

        # Clear current displayed messages
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if widget := item.widget():
                widget.deleteLater()

        # Re-add stretch after clearing
        self.content_layout.addStretch(1)

        # Re-add only those that pass the current filter
        for message in self.message_history:
            vad_conf = message.get("avg_vad_confidence", 1.0)
            if vad_conf >= min_vad:
                self._render_message_widget(message)

        self._scroll_to_bottom_smooth()

    def _connect_signals(self):
        self.signals._add_message.connect(self._do_add_message)
        self.signals._clear.connect(self._do_clear)
        self.signals._toggle_minimize.connect(self.toggle_minimize)

    def _fade_widget(
        self, widget, target_opacity: float = 0.38, duration_ms: int = 900
    ) -> None:
        """Smoothly animate opacity of a widget (used for older segment messages)"""
        if not hasattr(self, "_opacity_effects"):
            self._opacity_effects = {}
        if widget not in self._opacity_effects:
            effect = QGraphicsOpacityEffect()
            widget.setGraphicsEffect(effect)
            self._opacity_effects[widget] = effect
        else:
            effect = self._opacity_effects[widget]

        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(duration_ms)
        anim.setStartValue(effect.opacity())
        anim.setEndValue(target_opacity)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        # keep ref until finished
        self._active_animations.append(anim)
        anim.finished.connect(lambda a=anim: self._active_animations.remove(a))
        anim.start()

    def _do_clear(self) -> None:
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if widget := item.widget():
                widget.deleteLater()
        self.content_layout.addStretch(1)  # re-add stretch after clear
        self.history.clear()
        self.message_history.clear()

        # Reset segment grouping tracker
        self._last_rendered_segment = None

        # reset player
        self._player.stop()
        self._player.setSource(QUrl())  # clear source
        self._message_by_id.clear()  # also clean up stale ids
        self._widget_by_id.clear()

        # Reset segment tracking and opacity effects
        self._opacity_effects.clear()
        self._widgets_per_segment.clear()

    def position_window_top_right(self, margin_x: int = 0, margin_y: int = 0) -> None:
        screen = QApplication.primaryScreen()
        geometry = screen.availableGeometry() if screen else None
        if geometry is None:
            return
        x = geometry.right() - self.width() - margin_x
        y = geometry.top() + margin_y
        self.move(x, y)

    # --- PUBLIC API ---

    def add_message(
        self,
        translated_text: str,
        *,
        message_id: str | None = None,
        start_sec: float = 0.0,
        end_sec: float = 0.0,
        duration_sec: float = 0.0,
        source_text: str | None = None,
        segment_number: int | None = None,
        avg_vad_confidence: float | None = None,
        transcription_confidence: float | None = None,
        transcription_quality: str | None = None,
        translation_confidence: float | None = None,
        translation_quality: str | None = None,
        is_partial: bool | None = None,
        chunk_index: int | None = None,
        # removed avg_rms_linear and avg_rms_dbfs
        **kwargs,
    ) -> str:
        if not translated_text or not str(translated_text).strip():
            return ""

        mid = message_id or uuid.uuid4().hex

        subtitle_message: SubtitleMessage = {
            "id": mid,
            "translated_text": translated_text.strip(),
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            "duration_sec": round(duration_sec, 2),
            "source_text": (source_text or "").strip(),
        }

        if segment_number is not None:
            subtitle_message["segment_number"] = segment_number
        if avg_vad_confidence is not None:
            subtitle_message["avg_vad_confidence"] = avg_vad_confidence
        if transcription_confidence is not None:
            subtitle_message["transcription_confidence"] = transcription_confidence
        if transcription_quality is not None:
            subtitle_message["transcription_quality"] = transcription_quality
        if translation_confidence is not None:
            subtitle_message["translation_confidence"] = translation_confidence
        if translation_quality is not None:
            subtitle_message["translation_quality"] = translation_quality

        if is_partial is not None:
            subtitle_message["is_partial"] = is_partial
        if chunk_index is not None:
            subtitle_message["chunk_index"] = chunk_index

        # --- support rms and rms_label from kwargs ---
        if "rms" in kwargs and kwargs["rms"] is not None:
            subtitle_message["rms"] = kwargs["rms"]
        if "rms_label" in kwargs and kwargs["rms_label"]:
            subtitle_message["rms_label"] = kwargs["rms_label"]

        self.signals._add_message.emit(subtitle_message)
        return mid

    FuncType = Callable[..., SubtitleMessage | str | Awaitable[SubtitleMessage | str]]

    def add_task(self, func: FuncType, *args, **kwargs) -> None:
        """
        Add a sync/async callable for sequential execution.
        Its return/awaited value is displayed as a message.
        Displays a "Pending" row until processing starts, then a spinner + "Processing".
        """

        async def _wrapper() -> SubtitleMessage | str:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        coro = _wrapper()
        self.signals._enqueue_task.emit(coro, kwargs)

    def _on_enqueue_task(
        self, coro: Awaitable[SubtitleMessage | str], kwargs: dict
    ) -> None:
        loading_widget = QWidget()
        layout_pending = QHBoxLayout(loading_widget)
        layout_pending.setContentsMargins(6, 4, 6, 4)  # reduced from 8,6,8,6
        layout_pending.setAlignment(Qt.AlignmentFlag.AlignCenter)

        pending_text = QLabel("Pending")
        pending_text.setStyleSheet("color: #aaaaaa;")
        pending_text.setFont(
            QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 12)
        )  # 13 → 12
        layout_pending.addStretch()
        layout_pending.addWidget(pending_text)
        layout_pending.addStretch()

        self.content_layout.addWidget(loading_widget)
        QTimer.singleShot(
            0,
            lambda: self.scroll.verticalScrollBar().setValue(
                self.scroll.verticalScrollBar().maximum()
            ),
        )
        self.history.append("Pending")

        with self._task_lock:
            was_empty = len(self._pending_tasks) == 0
            self._pending_tasks.append((coro, loading_widget, layout_pending))
            if was_empty:
                self._process_next_task()

    def clear(self):
        """Remove all messages and reset transcript state."""
        self.signals._clear.emit()
        if hasattr(self, "_on_clear_callback") and self._on_clear_callback:
            try:
                self._on_clear_callback()
            except Exception:
                self.logger.exception("on_clear callback failed")

    def _setup_spinner_animation(
        self, spinner_label: QLabel, parent_widget: QWidget
    ) -> None:
        spinner_label.setStyleSheet("color: #ffff66;")
        spinner_label.setFont(
            QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 16)
        )  # 18 → 16
        spinner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠇"]
        current = 0

        def update_frame():
            nonlocal current
            spinner_label.setText(frames[current])
            current = (current + 1) % len(frames)

        timer = QTimer(self)
        timer.timeout.connect(update_frame)
        timer.start(80)
        parent_widget.setProperty("_spinner_timer", timer)
        parent_widget.destroyed.connect(timer.stop)

    def toggle_minimize(self):
        """Toggle overlay between full transcript and floating mini-bar."""
        if not self._is_minimized:
            self._original_size = self.size()
            self.setFixedHeight(64)
            self.content_area.layout().setEnabled(False)
            for i in reversed(range(self.content_area.layout().count())):
                item = self.content_area.layout().itemAt(i)
                if item.widget():
                    item.widget().hide()
            self.control_bar.layout().setContentsMargins(16, 12, 16, 12)
            self.status_label.setText("LIVE • MINIMIZED")
            if hasattr(self, "title_label"):
                self.title_label.show()
            self.min_btn.setText("□")
            self._is_minimized = True
        else:
            if self._original_size:
                self.setFixedSize(self._original_size)
            else:
                self.setFixedHeight(MINIMIZED_HEIGHT)
            for i in range(self.content_area.layout().count()):
                item = self.content_area.layout().itemAt(i)
                if item.widget():
                    item.widget().show()
            self.content_area.layout().setEnabled(True)
            self.control_bar.layout().setContentsMargins(12, 10, 12, 10)
            self.status_label.setText("LIVE")
            self.min_btn.setText("−")
            self._is_minimized = False

    def keyPressEvent(self, event):
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

    # --- INTERNAL ---

    def _render_message_widget(self, message: SubtitleMessage) -> None:
        """Reusable method to create and add a single message widget."""
        translated_text = message["translated_text"]
        source_text = message.get("source_text", "")
        duration_sec = message.get("duration_sec", 0.0)
        start_sec = message.get("start_sec", 0.0)
        end_sec = message.get("end_sec", 0.0)
        segment_number = message.get("segment_number")
        is_partial = message.get("is_partial", False)
        chunk_index = message.get("chunk_index")

        # ───────────────────────────────────────────────
        # Insert group separator when segment changes
        # ───────────────────────────────────────────────
        if segment_number is not None and segment_number != self._last_rendered_segment:
            self._insert_segment_separator(segment_number)
            self._last_rendered_segment = segment_number

        # Score-related fields (all optional)
        vad_conf = message.get("avg_vad_confidence")
        tr_conf = message.get("transcription_confidence")
        tl_conf = message.get("translation_confidence")
        rms_dbfs = message.get("avg_rms_dbfs")

        container = QWidget()
        container.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(4)
        container_layout.setContentsMargins(8, 5, 8, 5)

        # ───────────────────────────────────────────────
        # Row 1: status, duration, time range, buttons
        # ───────────────────────────────────────────────
        meta_layout1 = QHBoxLayout()
        meta_layout1.setContentsMargins(0, 0, 0, 0)
        meta_layout1.setSpacing(6)

        # Partial & chunk indicators (without segment number badge)
        status_parts = []
        if is_partial:
            status_parts.append("partial")
            if chunk_index is not None:
                display_idx = chunk_index + 1 if chunk_index >= 0 else chunk_index
                status_parts.append(f"chunk {display_idx}")

        if status_parts:
            status_text = " • ".join(status_parts)
            status_label = QLabel(status_text)
            status_label.setStyleSheet("""
                color: #ffbb66;
                font-style: italic;
                font-size: 11px;
            """)
            meta_layout1.addWidget(status_label)

        dur = QLabel(f"{duration_sec:.1f}s")
        dur.setStyleSheet("color: #aaa")
        dur.setFont(QFont("Segoe UI", 9))
        meta_layout1.addWidget(dur)

        time_range = QLabel(f"{start_sec:.1f} – {end_sec:.1f}")
        time_range.setStyleSheet("color: #888")
        time_range.setFont(QFont("Segoe UI", 9))
        meta_layout1.addWidget(time_range)

        # ── Copy buttons: translated and source text ────────────────────
        copy_tr = QToolButton()
        copy_tr.setIcon(QIcon.fromTheme("edit-copy"))
        # Fallback if theme icon not available:
        # copy_tr.setText("⎘")
        # copy_tr.setFont(QFont("Segoe UI", 11))  # optional sizing
        copy_tr.setToolTip("Copy translated text")
        copy_tr.setFixedSize(20, 20)
        copy_tr.setStyleSheet("""
            QToolButton {
                color: #cccccc;
                background: transparent;
                border: none;
                padding: 2px;
            }
            QToolButton:hover {
                background: rgba(180, 200, 255, 0.18);
                border-radius: 4px;
            }
            QToolButton:pressed {
                background: rgba(140, 180, 255, 0.35);
            }
        """)
        copy_tr.clicked.connect(
            lambda _: self._copy_to_clipboard(translated_text, "translation")
        )

        # Optional: slight visual distinction for source copy
        if source_text.strip():
            copy_src = QToolButton()
            copy_src.setIcon(QIcon.fromTheme("edit-copy"))
            # Alternative fallback:
            # copy_src.setText("⎘")
            copy_src.setToolTip("Copy original (source) text")
            copy_src.setFixedSize(20, 20)
            # Optional: slightly different idle color to distinguish
            copy_src.setStyleSheet(copy_tr.styleSheet().replace("#cccccc", "#bbbbff"))
            copy_src.clicked.connect(
                lambda _: self._copy_to_clipboard(source_text, "original")
            )
            meta_layout1.addWidget(copy_src)

        meta_layout1.addWidget(copy_tr)
        meta_layout1.addStretch()

        if segment_number is not None:
            play_btn = QToolButton()
            play_btn.setText("▶")
            play_btn.setFixedSize(26, 26)
            play_btn.setStyleSheet("""
                QToolButton {
                    background: rgba(80, 180, 120, 0.35);
                    color: white;
                    border-radius: 13px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QToolButton:hover { background: rgba(100, 220, 140, 0.65); }
                QToolButton:pressed { background: rgba(60, 160, 100, 0.85); }
            """)
            play_btn.setToolTip(f"Play segment #{segment_number}")
            play_btn.clicked.connect(
                lambda checked, num=segment_number: self._play_segment(message)
            )
            meta_layout1.addWidget(play_btn)

        container_layout.addLayout(meta_layout1)

        # ───────────────────────────────────────────────
        # Row 2: VAD  TR  TL  RMS    (no | separator, just spacing)
        # ───────────────────────────────────────────────
        meta_row2 = QWidget()
        meta_layout2 = QHBoxLayout(meta_row2)
        meta_layout2.setContentsMargins(0, 3, 0, 5)
        meta_layout2.setSpacing(18)  # ← increased spacing instead of explicit divider

        def conf_color(v: float | None) -> str:
            if v is None:
                return "#777"
            v = max(0.0, min(1.0, v))
            if v >= 0.95:
                return "#00ff9d"
            if v >= 0.85:
                return "#90ff50"
            if v >= 0.70:
                return "#ffd700"
            if v >= 0.50:
                return "#ffaa00"
            return "#ff4d4d"

        def get_rms_style(value: float | None) -> tuple[str, str]:
            """RMS color: red (quiet) → green (loud)"""
            if value is None:
                return "#777777", "N/A"

            if value < 0.004:
                return "#ff4444", "Very quiet"
            if value < 0.015:
                return "#ff7733", "Quiet"
            if value < 0.060:
                return "#ffaa44", "Soft"
            if value < 0.140:
                return "#ffdd44", "Normal"
            if value < 0.260:
                return "#bbff77", "Loud"
            if value < 0.380:
                return "#66ff99", "Raised"
            if value < 0.480:
                return "#00cc44", "Very Loud"
            return "#008f2f", "Extremely Loud"

        items = []

        if vad_conf is not None:
            lbl = QLabel(f"VAD {vad_conf:.0%}")
            lbl.setStyleSheet(f"color: {conf_color(vad_conf)}; font-weight: bold;")
            lbl.setFont(QFont("Segoe UI", 9))
            items.append(lbl)

        if tr_conf is not None:
            lbl = QLabel(f"TR {tr_conf:.0%}")
            lbl.setStyleSheet(f"color: {conf_color(tr_conf)}; font-weight: bold;")
            lbl.setFont(QFont("Segoe UI", 9))
            items.append(lbl)

        if tl_conf is not None:
            lbl = QLabel(f"TL {tl_conf:.0%}")
            lbl.setStyleSheet(f"color: {conf_color(tl_conf)}; font-weight: bold;")
            lbl.setFont(QFont("Segoe UI", 9))
            items.append(lbl)

        norm_rms = message.get("rms")
        norm_label = message.get("rms_label")
        if norm_rms is not None:
            color, fallback = get_rms_style(norm_rms)
            display_label = norm_label or fallback
            txt = f"RMS {norm_rms:.2f}"
            if display_label != "N/A":
                txt += f" • {display_label}"
            lbl = QLabel(txt)
            lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
            lbl.setFont(QFont("Segoe UI", 9))
            items.append(lbl)

        if items:
            for widget in items:
                meta_layout2.addWidget(widget)
            meta_layout2.addStretch()
            container_layout.addWidget(meta_row2)

        # ───────────────────────────────────────────────
        # Main text
        # ───────────────────────────────────────────────
        text_container = QWidget()
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(0, 2, 0, 2)
        text_layout.setSpacing(2)

        tr_label = QLabel(translated_text)
        tr_label.setWordWrap(True)
        tr_label.setStyleSheet("color: white;")
        tr_label.setFont(QFont("Segoe UI", 13))
        text_layout.addWidget(tr_label)

        if source_text and not self.hide_source_text:
            src = QLabel(source_text)
            src.setWordWrap(True)
            src.setStyleSheet("color: #b0b0ff;")
            src.setFont(QFont("Segoe UI", 10))
            text_layout.addWidget(src)

        container_layout.addWidget(text_container)

        container.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 rgba(30,35,60,0.7),
                    stop:1 rgba(50,60,90,0.6));
                border-radius: 8px;
                border: 1px solid rgba(90,120,160,0.4);
            }
            QWidget:hover {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 rgba(40,45,75,0.85),
                    stop:1 rgba(60,70,110,0.75));
            }
        """)

        # Attach opacity effect to every message container
        effect = QGraphicsOpacityEffect()
        effect.setOpacity(1.0)
        container.setGraphicsEffect(effect)
        self._opacity_effects[container] = effect

        self.content_layout.addWidget(container)

        mid = message["id"]
        self._message_by_id[mid] = message
        self._widget_by_id[mid] = container

        # ───────────────────────────────────────────────
        #   Segment-based fading logic
        # ───────────────────────────────────────────────
        segment_number = message.get("segment_number")
        if segment_number is not None:
            # Initialize list if first message for this segment
            if segment_number not in self._widgets_per_segment:
                self._widgets_per_segment[segment_number] = []

            # Fade ALL previous widgets for this segment
            for old_widget in self._widgets_per_segment[segment_number]:
                if old_widget in self._opacity_effects:
                    self._fade_widget(old_widget, target_opacity=0.50, duration_ms=800)

            # Add current widget to the list
            self._widgets_per_segment[segment_number].append(container)

            # Ensure newest one is always full opacity (in case it was faded before)
            effect = self._opacity_effects.get(container)
            if effect is not None:
                effect.setOpacity(1.0)

    def _insert_segment_separator(self, segment_number: int) -> None:
        """
        Visually separates different segment groups.
        Adds a header row with thicker spacing.
        """
        separator_container = QWidget()
        separator_layout = QVBoxLayout(separator_container)
        separator_layout.setContentsMargins(0, 12, 0, 4)
        separator_layout.setSpacing(4)

        # Horizontal line
        line = QWidget()
        line.setFixedHeight(2)
        line.setStyleSheet("""
            background: rgba(120, 160, 255, 0.5);
            border-radius: 1px;
        """)

        # Segment title
        title = QLabel(f"Segment #{segment_number}")
        title.setStyleSheet("""
            color: #8fd3ff;
            font-weight: bold;
            font-size: 12px;
            padding-left: 4px;
        """)

        separator_layout.addWidget(line)
        separator_layout.addWidget(title)

        self.content_layout.addWidget(separator_container)

    def _do_add_message(self, message: SubtitleMessage) -> None:
        mid = message["id"]

        # Replace existing message
        if mid in self._widget_by_id:
            old_widget = self._widget_by_id[mid]
            idx = self.content_layout.indexOf(old_widget)
            old_widget.deleteLater()
            self.content_layout.takeAt(idx)

            self._message_by_id[mid] = message
            self._render_message_widget(message)

            QTimer.singleShot(0, self._scroll_to_bottom_smooth)  # <-- ADD THIS
            return

        # Normal insert
        self._message_by_id[mid] = message
        self.message_history.append(message)
        self.history.append(message["translated_text"])

        vad_conf = message.get("avg_vad_confidence", 1.0)
        min_vad = (
            0.7
            if self.vad_high.isChecked()
            else 0.5
            if self.vad_med.isChecked()
            else 0.0
        )

        if vad_conf >= min_vad:
            self._render_message_widget(message)

        QTimer.singleShot(0, self._scroll_to_bottom_smooth)

    def _find_audio_file(
        self, message_id: str, chunk_index: int | None = None
    ) -> str | None:
        """
        Locate the .wav file in self.segments_dir that corresponds to this message.

        Expected filename pattern:  *_{last6 chars of id}_{chunk_index}.wav

        If chunk_index is provided → looks for exact _{suffix}_{chunk}.wav suffix
        If chunk_index is None     → looks for any file containing _{suffix}_*.wav
                                    and prefers the one with smallest / earliest chunk

        Returns absolute path if found, None otherwise.
        """
        if not message_id or len(message_id) < 6:
            self.logger.warning("[find_audio] Invalid message ID: %s", message_id)
            return None

        suffix = message_id[-6:]
        segment_dir = QDir(self.segments_dir)

        if not segment_dir.exists():
            self.logger.warning(
                "[find_audio] Directory not found: %s", self.segments_dir
            )
            return None

        # Build pattern
        if chunk_index is not None:
            # Exact match on chunk → highest confidence
            pattern = f"*_{suffix}_{chunk_index}.wav"
        else:
            # Fallback: any chunk for this utterance
            pattern = f"*_{suffix}_*.wav"

        candidates = segment_dir.entryList(
            [pattern],
            QDir.Filter.Files | QDir.Filter.Readable,
            QDir.SortFlag.Name,  # usually gives chronological-ish order
        )

        if not candidates:
            self.logger.debug(
                "[find_audio] No match for pattern '%s' in %s (id suffix: %s, chunk: %s)",
                pattern,
                self.segments_dir,
                suffix,
                chunk_index,
            )
            return None

        if len(candidates) > 1:
            self.logger.info(
                "[find_audio] Multiple matches for %s (chunk %s): %s → picking first",
                suffix,
                chunk_index,
                ", ".join(candidates),
            )
            # Alternative strategies you could add:
            # - sort by QFileInfo.lastModified() and take newest
            # - parse chunk number and take lowest/highest
            # For now: take first (usually Name sort ≈ creation order)

        chosen = candidates[0]
        full_path = segment_dir.absoluteFilePath(chosen)

        self.logger.debug("[find_audio] Selected: %s", chosen)
        return full_path

    def _play_segment(self, message: SubtitleMessage) -> None:
        """Play the audio segment corresponding to this subtitle message."""
        mid = message.get("id")
        chunk = message.get("chunk_index")

        if not mid:
            self.logger.warning("[play] Cannot play: no message id")
            return

        wav_path = self._find_audio_file(mid, chunk)

        if not wav_path:
            self.logger.info(
                "[play] No audio found for message %s (chunk %s)",
                mid[:8] + "…" if mid else "(no id)",
                chunk,
            )
            # Optional: show feedback in UI, e.g. flash status label
            return

        url = QUrl.fromLocalFile(wav_path)
        self._player.stop()
        self._player.setSource(url)
        self._player.play()

        self.logger.info(
            "[play] → %s  (id suffix: %s, chunk: %s)",
            Path(wav_path).name,
            mid[-6:],
            chunk,
        )

    def _scroll_to_bottom_smooth(self) -> None:
        scrollbar = self.scroll.verticalScrollBar()
        QApplication.processEvents()

        self._scroll_anim = QPropertyAnimation(
            scrollbar,
            b"value",
            self,
        )
        self._scroll_anim.setStartValue(scrollbar.value())
        self._scroll_anim.setEndValue(scrollbar.maximum())
        self._scroll_anim.setDuration(300)
        self._scroll_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._scroll_anim.start()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = (
                e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            e.accept()

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_pos)
            e.accept()

    def update_message(self, message_id: str, **updates) -> bool:
        """
        Update an existing message by id.
        Returns False if message doesn't exist.
        """
        msg = self._message_by_id.get(message_id)
        if not msg:
            return False

        msg.update({k: v for k, v in updates.items() if v is not None})
        self.signals._add_message.emit(msg)
        return True

    @classmethod
    def create(
        cls,
        app: QApplication | None = None,
        title: str | None = None,
        on_clear: Callable[[], None] | None = None,
        play_volume: float = PLAY_VOLUME,
        hide_source_text: bool = False,
    ) -> "LiveSubtitlesOverlay":
        """
        Instantiate the overlay, always positioned top-right, always-on-top, and thread-safe.
        - Creates QApplication if needed (or reuses existing)
        - Prevents quit on close/minimize
        - Positions window right
        - Handles Ctrl+C gracefully
        - Thread-safe .add_message()
        - Optional custom title
        - Optional on_clear callback
        """
        app = app or QApplication.instance() or QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(False)

        def _quit_on_sigint(sig, frame):
            app.quit()

        signal.signal(signal.SIGINT, _quit_on_sigint)

        # Pass through the optional callback
        overlay = cls(
            title=title,
            on_clear=on_clear,
            play_volume=play_volume,
            hide_source_text=hide_source_text,
        )
        overlay.show()
        overlay.raise_()
        overlay.activateWindow()

        def _do_center():
            overlay.updateGeometry()
            overlay.position_window_top_right(margin_x=30, margin_y=0)
            overlay.logger.info("[green]Overlay positioned top-right[/]")

        QTimer.singleShot(50, _do_center)
        return overlay
