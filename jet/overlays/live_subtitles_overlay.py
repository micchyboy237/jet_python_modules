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
    QFile,
    QObject,
    QPoint,
    QPropertyAnimation,
    Qt,
    QTimer,
    QUrl,
    pyqtSignal,
)
from PyQt6.QtGui import QFont
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QApplication,
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
    avg_vad_confidence: NotRequired[float]
    transcription_confidence: NotRequired[float]
    transcription_quality: NotRequired[str]
    translation_confidence: NotRequired[float]
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
        play_volume: float = 0.5,
        segments_dir: str | None = None,
    ):
        super().__init__(parent)
        self.logger = _setup_logging()
        self._on_clear_callback = on_clear
        self.signals = _Signals()
        self.history = []
        self.title = title
        self.message_history: list[SubtitleMessage] = []
        self._task_lock = threading.Lock()  # for thread-safety with add_task

        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 190); border-radius: 12px;")
        self.setFixedSize(450, 900)

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

    def _do_clear(self) -> None:
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if widget := item.widget():
                widget.deleteLater()
        self.history.clear()
        self.message_history.clear()

    def position_window_top_right(self, margin: int = 0) -> None:
        screen = QApplication.primaryScreen().availableGeometry()
        x = screen.right() - self.width() + 1
        y = screen.top() - 1
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
    ) -> str:
        if not translated_text or not str(translated_text).strip():
            return ""

        mid = message_id or uuid.uuid4().hex

        subtitle_message: SubtitleMessage = {
            "id": mid,
            "translated_text": translated_text.strip(),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": duration_sec,
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
                self.setFixedHeight(600)
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
        """Reusable method to create and add a single message widget.
        Extracted from _do_add_message to support filtering without duplication."""
        translated_text = message["translated_text"]
        source_text = message.get("source_text", "")
        duration_sec = message.get("duration_sec", 0.0)
        start_sec = message.get("start_sec", 0.0)
        end_sec = message.get("end_sec", 0.0)
        segment_number = message.get("segment_number")
        vad_conf = message.get("avg_vad_confidence")
        tr_conf = message.get("transcription_confidence")
        tr_quality = message.get("transcription_quality")
        tl_conf = message.get("translation_confidence")
        tl_quality = message.get("translation_quality")

        container = QWidget()
        container.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(4)
        container_layout.setContentsMargins(8, 5, 8, 5)

        # Metadata row
        meta_widget = QWidget()
        meta_layout = QHBoxLayout(meta_widget)
        meta_layout.setContentsMargins(0, 0, 0, 0)
        meta_layout.setSpacing(8)

        # Segment number
        if segment_number is not None:
            seg = QLabel(f"#{segment_number}")
            seg.setStyleSheet("""
                color: #aaffaa;
                background: rgba(0, 120, 0, 0.45);
                border-radius: 5px;
                padding: 2px 8px;
                font-weight: bold;
            """)
            seg.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            meta_layout.addWidget(seg)

        dur = QLabel(f"{duration_sec:.1f}s")
        dur.setStyleSheet("color: #b0d0ff;")
        dur.setFont(QFont("Segoe UI", 9))
        meta_layout.addWidget(dur)

        time_range = QLabel(f"{start_sec:.1f} – {end_sec:.1f}")
        time_range.setStyleSheet("color: #90b0d0; font-size: 9pt;")
        meta_layout.addWidget(time_range)

        # Confidence / quality labels (existing)
        quality_colors = {
            "Very High": "#4ade80",
            "High": "#a3e635",
            "Good": "#fbbf24",
            "Medium": "#fb923c",
            "Low": "#f87171",
            "N/A": "#aaaaaa",
        }

        def get_quality_style(q: str | None) -> str:
            color = quality_colors.get(q or "N/A", "#aaaaaa")
            return f"color: {color}; font-size:9pt; font-weight:bold;"

        def conf_color(v: float | None) -> str:
            if v is None:
                return "#aaaaaa"
            return "#4ade80" if v >= 0.90 else "#fbbf24" if v >= 0.75 else "#f87171"

        if vad_conf is not None:
            vad = QLabel(f"VAD {vad_conf:.0%}")
            vad.setStyleSheet(f"color:{conf_color(vad_conf)}; font-weight:bold;")
            vad.setFont(QFont("Segoe UI", 9))
            vad.setToolTip("Average VAD confidence during voice detection")
            meta_layout.addWidget(vad)

        if tr_conf is not None:
            trc = QLabel(f"Tr {tr_conf:.0%}")
            trc.setStyleSheet(f"color:{conf_color(tr_conf)}; font-weight:bold;")
            trc.setFont(QFont("Segoe UI", 9))
            trc.setToolTip("Transcription confidence (exp(avg logprob))")
            meta_layout.addWidget(trc)

            if tr_quality:
                trq = QLabel(tr_quality)
                trq.setStyleSheet(get_quality_style(tr_quality))
                trq.setToolTip("Transcription quality assessment")
                meta_layout.addWidget(trq)

        if tl_conf is not None:
            tl_label = QLabel(f"TL {tl_conf:.0%}")
            tl_label.setStyleSheet(f"color:{conf_color(tl_conf)}; font-weight:bold;")
            tl_label.setFont(QFont("Segoe UI", 9))
            tl_label.setToolTip(
                "Translation confidence (normalized 0–1, higher = better)"
            )
            meta_layout.addWidget(tl_label)

            if tl_quality:
                tlq = QLabel(tl_quality)
                tlq.setStyleSheet(get_quality_style(tl_quality))
                tlq.setToolTip("Translation quality assessment")
                meta_layout.addWidget(tlq)

        meta_layout.addStretch()

        # ─── Play button ────────────────────────────────────────────────────────
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
                QToolButton:hover {
                    background: rgba(100, 220, 140, 0.65);
                }
                QToolButton:pressed {
                    background: rgba(60, 160, 100, 0.85);
                }
            """)
            play_btn.setToolTip(f"Play segment #{segment_number:04d}")
            play_btn.clicked.connect(
                lambda checked, num=segment_number: self._play_segment(num)
            )
            meta_layout.addWidget(play_btn)

        container_layout.addWidget(meta_widget)

        # Text area
        text_container = QWidget()
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(0, 2, 0, 2)
        text_layout.setSpacing(2)

        tr_label = QLabel(translated_text)
        tr_label.setWordWrap(True)
        tr_label.setStyleSheet("color: white;")
        tr_label.setFont(QFont("Segoe UI", 13))
        text_layout.addWidget(tr_label)

        if source_text:
            src = QLabel(source_text)
            src.setWordWrap(True)
            src.setStyleSheet("color: #d0d0ff; font-style: italic;")
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

        self.content_layout.addWidget(container)

        mid = message["id"]

        self._message_by_id[mid] = message
        self._widget_by_id[mid] = container
        self.content_layout.addWidget(container)

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

    def _play_segment(self, segment_num: int) -> None:
        """Play the audio file for the given segment number."""
        segment_dir = QDir(self.segments_dir).filePath(f"segment_{segment_num:04d}")
        wav_path = QDir(segment_dir).filePath("sound.wav")

        if not QDir(segment_dir).exists() or not QFile(wav_path).exists():
            self.logger.warning("[play] Audio not found: %s", wav_path)
            return

        url = QUrl.fromLocalFile(wav_path)
        self._player.stop()  # stop any previous playback
        self._player.setSource(url)
        self._player.play()
        self.logger.info("[play] Started segment_%04d → %s", segment_num, wav_path)

    def _scroll_to_bottom_smooth(self) -> None:
        scrollbar = self.scroll.verticalScrollBar()
        QApplication.processEvents()
        QPropertyAnimation(
            scrollbar,
            b"value",
            self,
            startValue=scrollbar.value(),
            endValue=scrollbar.maximum(),
            duration=300,
            easingCurve=QEasingCurve.Type.OutCubic,
        ).start()

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
        play_volume: float = 0.5,
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
        overlay = cls(title=title, on_clear=on_clear, play_volume=play_volume)
        overlay.show()
        overlay.raise_()
        overlay.activateWindow()

        def _do_center():
            overlay.updateGeometry()
            overlay.position_window_top_right(margin=30)
            overlay.logger.info("[green]Overlay positioned top-right[/]")

        QTimer.singleShot(50, _do_center)
        return overlay
