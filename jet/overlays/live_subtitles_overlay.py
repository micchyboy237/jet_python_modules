# live_subtitles_overlay.py
# Usage Example:
# overlay = LiveSubtitlesOverlay.create()
# overlay.add_message("Your text here")   # Exactly what you want!

import sys
import signal
import logging
import time
import threading
from typing import Optional, Awaitable, TypedDict, Callable, Union, NotRequired
import asyncio
from concurrent.futures import Future

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QObject, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont

from rich.logging import RichHandler


def _setup_logging():
    logger = logging.getLogger("LiveSubtitlesOverlay")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True, markup=True, show_path=False)
        handler.setFormatter(logging.Formatter("[bold magenta]Subtitle[/] → %(message)s"))
        logger.addHandler(handler)
    return logger


class SubtitleMessage(TypedDict):
    translated_text: str
    start_sec: float
    end_sec: float
    duration_sec: float
    source_text: str
    segment_number: NotRequired[int]
    avg_vad_confidence: NotRequired[float]
    transcription_confidence: NotRequired[float]
    transcription_quality: NotRequired[str]
    translation_confidence: NotRequired[float]      # normalized 0.0–1.0 confidence
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

    def __init__(self, parent=None, title: Optional[str] = None):
        super().__init__(parent)
        self.logger = _setup_logging()
        self.signals = _Signals()
        self.history = []
        self.title = title
        self.message_history: list[SubtitleMessage] = []
        self._task_lock = threading.Lock()  # for thread-safety with add_task

        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 190); border-radius: 12px;")
        self.setFixedSize(560, 900)

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

        self._pending_tasks: list[tuple[Awaitable[Union[SubtitleMessage, str]], QWidget, QHBoxLayout]] = []

        # Initial message
        self.logger.info("[green]Ready – use .add_message('text')[/]")
        self._process_next_task()

    def _start_async_loop(self):
        def run_loop():
            self._async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._async_loop)
            self._async_loop.run_forever()
        self._async_loop_thread = threading.Thread(target=run_loop, daemon=True, name="OverlayAsyncLoop")
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
        processing_text.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 13))  # 15 → 13
        current_layout.addStretch()
        current_layout.addWidget(spinner)
        current_layout.addWidget(processing_text)
        current_layout.addStretch()

        def run_in_thread() -> tuple[Union[SubtitleMessage, str], QWidget]:
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
                **({k: v for k, v in {
                    "segment_number": result.get("segment_number"),
                    "avg_vad_confidence": result.get("avg_vad_confidence"),
                    "transcription_confidence": result.get("transcription_confidence"),
                }.items() if v is not None}),
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
            new_label.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 15))
            self.content_layout.insertWidget(idx, new_label)
            self.history.append(text)

        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()))

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
        main = QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)          # reduced from 12
        main.setSpacing(4)                           # reduced from 8

        # Top bar: title, status, buttons
        self.control_bar = QHBoxLayout()
        self.control_bar.setContentsMargins(8, 6, 8, 6)   # reduced from 10,8,10,8
        self.control_bar.setSpacing(8)                   # reduced from 10

        if self.title:
            self.title_label = QLabel(self.title)
            self.title_label.setStyleSheet("color: #ffffff; font-size: 14px; font-weight: bold;")  # 16 → 14
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
        """)  # reduced padding & size
        self.control_bar.addWidget(self.status_label)
        self.control_bar.addStretch()

        self.min_btn = QPushButton("Minimize")
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedSize(28, 28)                # smaller buttons
        clear_btn.setStyleSheet("""
            QPushButton {
                background: rgba(180, 100, 100, 0.25);
                color: white;
                border-radius: 14px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover { background: rgba(255, 80, 80, 0.5); }
        """)
        clear_btn.clicked.connect(self.clear)

        self.min_btn.setFixedSize(28, 28)
        self.min_btn.setStyleSheet("""
            QPushButton {
                background: rgba(100, 180, 255, 0.25);
                color: white;
                border-radius: 14px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover { background: rgba(100, 180, 255, 0.5); }
        """)

        close_btn = QPushButton("×")
        close_btn.setFixedSize(28, 28)
        close_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 80, 80, 0.7);
                color: white;
                border-radius: 14px;
                font-weight: bold;
                font-size: 15px;
            }
            QPushButton:hover { background: rgba(255, 50, 50, 1); }
        """)
        close_btn.clicked.connect(QApplication.quit)

        self.control_bar.addWidget(clear_btn)
        self.control_bar.addWidget(self.min_btn)
        self.control_bar.addWidget(close_btn)

        self.control_bar_widget = QWidget()
        self.control_bar_widget.setLayout(self.control_bar)
        self.control_bar_widget.setCursor(Qt.CursorShape.OpenHandCursor)
        main.addWidget(self.control_bar_widget)

        # Content area (scrollable)
        self.content_area = QVBoxLayout()
        self.content_area.setSpacing(4)              # reduced from 6

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.content_layout.setSpacing(4)            # reduced from 6
        self.content_layout.setContentsMargins(6, 6, 6, 6)  # reduced from 8
        self.scroll.setWidget(self.content)

        self.content_area.addWidget(self.scroll, stretch=1)
        main.addLayout(self.content_area, stretch=1)

        self.min_btn.clicked.connect(self.toggle_minimize)
        self._is_minimized = False
        self._original_size = None

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
        start_sec: float = 0.0,
        end_sec: float = 0.0,
        duration_sec: float = 0.0,
        source_text: Optional[str] = None,
        segment_number: Optional[int] = None,
        avg_vad_confidence: Optional[float] = None,
        transcription_confidence: Optional[float] = None,
        transcription_quality: Optional[str] = None,
        translation_confidence: Optional[float] = None,   # new: 0.0–1.0
        translation_quality: Optional[str] = None,
    ) -> None:
        """Add a message thread-safely"""
        if not translated_text or not str(translated_text).strip():
            return

        subtitle_message: SubtitleMessage = {
            "translated_text": str(translated_text).strip(),
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

    FuncType = Callable[..., Union[SubtitleMessage, str, Awaitable[Union[SubtitleMessage, str]]]]

    def add_task(self, func: FuncType, *args, **kwargs) -> None:
        """
        Add a sync/async callable for sequential execution.
        Its return/awaited value is displayed as a message.
        Displays a "Pending" row until processing starts, then a spinner + "Processing".
        """
        async def _wrapper() -> Union[SubtitleMessage, str]:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        coro = _wrapper()
        self.signals._enqueue_task.emit(coro, kwargs)

    def _on_enqueue_task(self, coro: Awaitable[Union[SubtitleMessage, str]], kwargs: dict) -> None:
        loading_widget = QWidget()
        layout_pending = QHBoxLayout(loading_widget)
        layout_pending.setContentsMargins(6, 4, 6, 4)    # reduced from 8,6,8,6
        layout_pending.setAlignment(Qt.AlignmentFlag.AlignCenter)

        pending_text = QLabel("Pending")
        pending_text.setStyleSheet("color: #aaaaaa;")
        pending_text.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 12))  # 13 → 12
        layout_pending.addStretch()
        layout_pending.addWidget(pending_text)
        layout_pending.addStretch()

        self.content_layout.addWidget(loading_widget)
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()))
        self.history.append("Pending")

        with self._task_lock:
            was_empty = len(self._pending_tasks) == 0
            self._pending_tasks.append((coro, loading_widget, layout_pending))
            if was_empty:
                self._process_next_task()

    def clear(self):
        """Remove all messages and reset transcript state."""
        self.signals._clear.emit()

    def _setup_spinner_animation(self, spinner_label: QLabel, parent_widget: QWidget) -> None:
        spinner_label.setStyleSheet("color: #ffff66;")
        spinner_label.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 16))  # 18 → 16
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
            if hasattr(self, 'title_label'):
                self.title_label.show()
            self.min_btn.setText("Restore")
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
            self.min_btn.setText("Minimize")
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

    def _do_add_message(self, message: SubtitleMessage) -> None:
        translated_text = message["translated_text"]
        source_text     = message.get("source_text", "")
        start_sec       = message.get("start_sec", 0.0)
        end_sec         = message.get("end_sec", 0.0)
        duration_sec    = message.get("duration_sec", 0.0)
        segment_number  = message.get("segment_number")
        vad_conf        = message.get("avg_vad_confidence")
        tr_conf         = message.get("transcription_confidence")
        tr_quality      = message.get("transcription_quality")
        tl_conf         = message.get("translation_confidence")   # new
        tl_quality      = message.get("translation_quality")

        self.history.append(translated_text)
        self.message_history.append(message)

        container = QWidget()
        container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        # Main vertical layout: metadata on top, text below
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(4)
        container_layout.setContentsMargins(8, 5, 8, 5)

        # ── 1. Compact metadata row(s) ───────────────────────────────────────
        meta_widget = QWidget()
        meta_layout = QHBoxLayout(meta_widget)
        meta_layout.setContentsMargins(0, 0, 0, 0)
        meta_layout.setSpacing(8)

        # Segment number pill
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

        # Duration
        dur = QLabel(f"{duration_sec:.1f}s")
        dur.setStyleSheet("color: #b0d0ff;")
        dur.setFont(QFont("Segoe UI", 9))
        meta_layout.addWidget(dur)

        # Time range
        time_range = QLabel(f"{start_sec:.1f} – {end_sec:.1f}")
        time_range.setStyleSheet("color: #90b0d0; font-size: 9pt;")
        meta_layout.addWidget(time_range)

        # Confidence / Quality items (horizontal, compact)
        quality_colors = {
            "Very High": "#4ade80",
            "High":      "#a3e635",
            "Good":      "#fbbf24",
            "Medium":    "#fb923c",
            "Low":       "#f87171",
            "N/A":       "#aaaaaa",
        }

        def get_quality_style(q: str | None) -> str:
            color = quality_colors.get(q or "N/A", "#aaaaaa")
            return f"color: {color}; font-size:9pt; font-weight:bold;"

        def conf_color(v: float | None) -> str:
            if v is None: return "#aaaaaa"
            val = v  # for normalized confidence, higher is already better
            return "#4ade80" if val >= 0.90 else "#fbbf24" if val >= 0.75 else "#f87171"

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
            tl_text = f"TL {tl_conf:.0%}"
            tl_label = QLabel(tl_text)
            tl_label.setStyleSheet(f"color:{conf_color(tl_conf)}; font-weight:bold;")
            tl_label.setFont(QFont("Segoe UI", 9))
            tl_label.setToolTip("Translation confidence (normalized 0–1, higher = better)")
            meta_layout.addWidget(tl_label)
            if tl_quality:
                tlq = QLabel(tl_quality)
                tlq.setStyleSheet(get_quality_style(tl_quality))
                tlq.setToolTip("Translation quality assessment")
                meta_layout.addWidget(tlq)

        # Stretch to push content left
        meta_layout.addStretch()
        container_layout.addWidget(meta_widget)

        # ── 2. Full-width text area below ─────────────────────────────────────
        text_container = QWidget()
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(0, 2, 0, 2)
        text_layout.setSpacing(2)

        # Translated text (main line)
        tr_label = QLabel(translated_text)
        tr_label.setWordWrap(True)
        tr_label.setStyleSheet("color: white;")
        tr_label.setFont(QFont("Segoe UI", 13))
        text_layout.addWidget(tr_label)

        # Source text (if present)
        if source_text:
            src = QLabel(source_text)
            src.setWordWrap(True)
            src.setStyleSheet("color: #d0d0ff; font-style: italic;")
            src.setFont(QFont("Segoe UI", 10))
            text_layout.addWidget(src)

        container_layout.addWidget(text_container)

        # Container background + hover
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
        QTimer.singleShot(0, self._scroll_to_bottom_smooth)


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
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_pos)
            e.accept()

    @classmethod
    def create(cls, app: Optional[QApplication] = None, title: Optional[str] = None) -> 'LiveSubtitlesOverlay':
        """
        Instantiate the overlay, always positioned top-right, always-on-top, and thread-safe.
        - Creates QApplication if needed
        - Prevents quit on close/minimize
        - Positions window right
        - Handles Ctrl+C gracefully
        - Thread-safe .add_message()
        - Optional custom title
        """
        app = app or QApplication.instance() or QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(False)

        def _quit_on_sigint(sig, frame):
            app.quit()
        signal.signal(signal.SIGINT, _quit_on_sigint)

        overlay = cls(title=title)
        overlay.show()
        overlay.raise_()
        overlay.activateWindow()

        def _do_center():
            overlay.updateGeometry()
            overlay.position_window_top_right(margin=30)
            overlay.logger.info("[green]Overlay positioned top-right[/]")

        QTimer.singleShot(50, _do_center)
        return overlay
