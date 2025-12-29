# subtitle_overlay.py
# One-liner: overlay = LiveSubtitlesOverlay.create()
# Then:      overlay.add_message("Your text here")   ← exactly what you want

import sys
import signal
import logging
import time
import threading
from typing import Optional, Awaitable, TypedDict, Callable, Union
import asyncio
from concurrent.futures import Future

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea
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
    translated_text: str          # Displayed prominently (e.g., English/target language)
    start_sec: float
    end_sec: float
    duration_sec: float
    source_text: str              # Original text (e.g., Japanese/source language)


class _Signals(QObject):
    # Emit the full SubtitleMessage object as one argument
    _add_message = pyqtSignal(object)  # object accepts dict or TypedDict
    _clear = pyqtSignal()
    _toggle_minimize = pyqtSignal()
    _enqueue_task = pyqtSignal(object, dict)  # (coro: Awaitable, kwargs: dict)


class LiveSubtitlesOverlay(QWidget):
    """
    Modern, thread-safe, live overlay for displaying subtitles, transcripts, or status messages.

    Usage:
        overlay = LiveSubtitlesOverlay.create()
        overlay.add_message("Hello world!")           # ← easy, always safe
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
        # Lock to make concurrent add_task calls thread-safe
        self._task_lock = threading.Lock()

        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 190); border-radius: 16px;")
        self.setFixedSize(720, 900)  # narrower width, taller height

        self._drag_pos = QPoint()
        self._build_ui()
        self._connect_signals()
        self.signals._enqueue_task.connect(self._on_enqueue_task)

        # ------------------------------------------------------------------
        # ASYNCIO/TASK INTEGRATION (Always sequential)
        # ------------------------------------------------------------------
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._async_loop = None
        self._async_loop_thread = None
        self._start_async_loop()

        self._pending_tasks: list[tuple[Awaitable[Union[SubtitleMessage, str]], QWidget, QHBoxLayout]] = []

        self.add_message(
            translated_text="Live Subtitle Overlay • Ready",
            source_text="",
        )
        self.logger.info("[green]Ready – use .add_message('text')[/]")
        self._process_next_task()

    def _start_async_loop(self):
        def run_loop():
            self._async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._async_loop)
            self._async_loop.run_forever()
        self._async_loop_thread = threading.Thread(target=run_loop, daemon=True, name="OverlayAsyncLoop")
        self._async_loop_thread.start()
        time.sleep(0.05)  # give loop time to start

    def _process_next_task(self) -> None:
        """Start the next queued async/sync task if available (runs sequentially)."""
        if not self._pending_tasks:
            return

        coro, loading_widget, current_layout = self._pending_tasks[0]

        # Clear pending safely using the stored layout
        while current_layout.count():
            item = current_layout.takeAt(0)
            if w := item.widget():
                w.deleteLater()
        # Do **not** do loading_widget.setLayout(...) here — it already has one!

        spinner = QLabel()
        self._setup_spinner_animation(spinner, loading_widget)

        processing_text = QLabel("Processing")
        processing_text.setStyleSheet("color: #4da6ff;")
        processing_text.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 20))

        # Reuse the existing layout
        current_layout.addStretch()
        current_layout.addWidget(spinner)
        current_layout.addWidget(processing_text)
        current_layout.addStretch()

        def run_in_thread() -> tuple[Union[SubtitleMessage, str], QWidget]:
            try:
                future = asyncio.run_coroutine_threadsafe(coro, self._async_loop)
                result = future.result()  # blocks until coro completes or raises
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
        """Handles completion of an add_task call – MUST run in main thread."""
        if future.cancelled():
            return

        # Remove the finished task from pending queue
        if self._pending_tasks:
            self._pending_tasks.pop(0)

        try:
            result, loading_widget = future.result()
        except Exception as e:
            result = f"[Error] {e}"

        # All UI operations below must be done in main thread.
        # Since this method is called via QTimer from main thread, we are safe.
        if not loading_widget or not loading_widget.parent():
            if self._pending_tasks:
                self._process_next_task()
            return

        # Stop spinner safely (timer belongs to main thread and is parented to self)
        timer = loading_widget.property("_spinner_timer")
        if timer and isinstance(timer, QTimer) and timer.isActive():
            timer.stop()
            timer.deleteLater()  # optional cleanup

        idx = self.content_layout.indexOf(loading_widget)
        if idx == -1:
            if self._pending_tasks:
                self._process_next_task()
            return

        self.content_layout.takeAt(idx)
        loading_widget.deleteLater()

        # Handle rich SubtitleMessage or fallback to simple string
        if isinstance(result, dict) and "translated_text" in result:
            subtitle_message: SubtitleMessage = {
                "translated_text": str(result.get("translated_text", "")),
                "start_sec": float(result.get("start_sec", 0.0)),
                "end_sec": float(result.get("end_sec", 0.0)),
                "duration_sec": float(result.get("duration_sec", 0.0)),
                "source_text": str(result.get("source_text", "")),
            }
            self.message_history.append(subtitle_message)
            self.history.append(subtitle_message["translated_text"])
            self.signals._add_message.emit(subtitle_message)
        else:
            # Backward-compatible simple string result
            text = str(result)
            new_label = QLabel(text)
            new_label.setWordWrap(True)
            new_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            new_label.setStyleSheet("color: white; padding: 4px;")
            new_label.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 20))
            self.content_layout.insertWidget(idx, new_label)
            self.history.append(text)

        # Smooth scroll in main thread
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()))

        # Continue with next task
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
        main.setContentsMargins(16, 16, 16, 16)
        main.setSpacing(10)

        # --- Top control bar (title, status, buttons) ---
        self.control_bar = QHBoxLayout()
        self.control_bar.setContentsMargins(12, 10, 12, 10)
        self.control_bar.setSpacing(12)

        if self.title:
            self.title_label = QLabel(self.title)
            self.title_label.setStyleSheet("color: #ffffff; font-size: 18px; font-weight: bold;")
            self.control_bar.addWidget(self.title_label)

        self.status_label = QLabel("LIVE")
        self.status_label.setStyleSheet("""
            color: #ff4444;
            background: rgba(255, 50, 50, 0.25);
            border: 1px solid rgba(255, 100, 100, 0.4);
            border-radius: 10px;
            padding: 4px 14px;
            font-weight: bold;
            font-size: 15px;
        """)
        self.control_bar.addWidget(self.status_label)
        self.control_bar.addStretch()

        self.min_btn = QPushButton("Minimize")
        self.min_btn.setFixedSize(38, 38)
        self.min_btn.setStyleSheet("""
            QPushButton {
                background: rgba(100, 180, 255, 0.25);
                color: white;
                border-radius: 19px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover { background: rgba(100, 180, 255, 0.5); }
        """)

        close_btn = QPushButton("×")
        close_btn.setFixedSize(38, 38)
        close_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 80, 80, 0.7);
                color: white;
                border-radius: 19px;
                font-weight: bold;
                font-size: 18px;
            }
            QPushButton:hover { background: rgba(255, 50, 50, 1); }
        """)
        close_btn.clicked.connect(QApplication.quit)

        self.control_bar.addWidget(self.min_btn)
        self.control_bar.addWidget(close_btn)

        self.control_bar_widget = QWidget()
        self.control_bar_widget.setLayout(self.control_bar)
        self.control_bar_widget.setCursor(Qt.CursorShape.OpenHandCursor)
        main.addWidget(self.control_bar_widget)

        # --- Content area (scrollable only) ---
        self.content_area = QVBoxLayout()
        self.content_area.setSpacing(8)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.content_layout.setSpacing(8)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.scroll.setWidget(self.content)

        self.content_area.addWidget(self.scroll, stretch=1)

        main.addLayout(self.content_area, stretch=1)

        self.min_btn.clicked.connect(self.toggle_minimize)
        self._is_minimized = False
        self._original_size = None

    def _connect_signals(self):
        self.signals._add_message.connect(self._do_add_message)
        self.signals._clear.connect(self.clear)
        self.signals._toggle_minimize.connect(self.toggle_minimize)

    def position_window_top_right(self, margin: int = 0) -> None:
        """Position the overlay flush against the top-right corner of the primary screen."""
        screen = QApplication.primaryScreen().availableGeometry()
        x = screen.right() - self.width() + 1  # +1 to account for window border/shadow
        y = screen.top() - 1                   # flush to top
        self.move(x, y)

    # --- PUBLIC API ---
    def add_message(
        self,
        translated_text: str,
        *,
        start_sec: float = 0.0,
        end_sec: float = 0.0,
        duration_sec: float = 0.0,
        source_text: str = "",
    ) -> None:
        """Thread-safe add_message – displays translated_text with optional source_text below."""
        if not translated_text or not str(translated_text).strip():
            return

        subtitle_message = {
            "translated_text": str(translated_text).strip(),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": duration_sec,
            "source_text": source_text.strip(),
        }
        self.signals._add_message.emit(subtitle_message)

    from typing import Awaitable
    FuncType = Callable[..., Union[SubtitleMessage, str, Awaitable[SubtitleMessage | str]]]

    def add_task(self, func: FuncType, *args, **kwargs) -> None:
        """
        Add a sync/async callable for sequential execution.
        Its return/awaited value is displayed as a message.
        Displays a "Pending" row until processing starts, then replaces it with spinner + "Processing".
        """
        async def _wrapper() -> Union[SubtitleMessage, str]:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        coro = _wrapper()  # call it here to get the coroutine object
        self.signals._enqueue_task.emit(coro, kwargs)

    def _on_enqueue_task(self, coro: Awaitable[Union[SubtitleMessage, str]], kwargs: dict) -> None:
        """Main-thread only: create Pending UI + queue real task"""
        loading_widget = QWidget()
        layout_pending = QHBoxLayout(loading_widget)
        layout_pending.setContentsMargins(10, 8, 10, 8)
        layout_pending.setAlignment(Qt.AlignmentFlag.AlignCenter)

        pending_text = QLabel("Pending")
        pending_text.setStyleSheet("color: #aaaaaa;")
        pending_text.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 20))
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
        """Remove all messages and reset transcript summary."""
        self.signals._clear.emit()

    def _setup_spinner_animation(self, spinner_label: QLabel, parent_widget: QWidget) -> None:

        spinner_label.setStyleSheet("color: #ffff66;")
        spinner_label.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 28))
        spinner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠇"]
        current = 0

        def update_frame():
            nonlocal current
            spinner_label.setText(frames[current])
            current = (current + 1) % len(frames)

        # FIXED: Timer must be created in the GUI thread and parented to the overlay
        timer = QTimer(self)  # parented to the main QWidget (GUI thread)
        timer.timeout.connect(update_frame)
        timer.start(80)

        # Store the timer on the parent_widget so we can stop it later
        parent_widget.setProperty("_spinner_timer", timer)

        # Ensure timer stops if the widget is destroyed
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
        """Global hotkeys: Ctrl+M = toggle minimize, Ctrl+Q/Esc = quit"""
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

    # --- INTERNAL ----
    def _do_add_message(self, message: SubtitleMessage) -> None:
        """Internal slot – receives the full SubtitleMessage dict."""
        translated_text = message["translated_text"]
        source_text = message.get("source_text", "")
        start_sec = message.get("start_sec", 0.0)
        end_sec = message.get("end_sec", 0.0)
        duration_sec = message.get("duration_sec", 0.0)

        self.history.append(translated_text)
        self.message_history.append(message)

        # Ultra-compact single-line container
        container = QWidget()
        main_layout = QHBoxLayout(container)
        main_layout.setSpacing(10)                   # reduced from 14
        main_layout.setContentsMargins(12, 5, 12, 5)  # minimal vertical/horizontal margins

        # Translated text – tight padding
        trans_label = QLabel(translated_text)
        trans_label.setWordWrap(True)
        trans_label.setStyleSheet(
            "color: #ffffff; background: rgba(255, 255, 255, 0.08); "
            "border-radius: 6px; padding: 6px 10px;"  # reduced from 8px 12px
        )
        trans_label.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 16, QFont.Weight.Bold))

        # Source text – minimal padding
        if source_text:
            src_label = QLabel(source_text)
            src_label.setWordWrap(True)
            src_label.setStyleSheet(
                "color: #aaccff; font-style: italic; background: rgba(100, 140, 255, 0.12); "
                "border-radius: 6px; padding: 5px 8px;"   # reduced from 6px 10px
            )
            src_label.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 12))

            main_layout.addWidget(trans_label, stretch=6)
            main_layout.addWidget(src_label, stretch=4)
        else:
            main_layout.addWidget(trans_label, stretch=1)

        # Timing info – super tight
        timing_layout = QVBoxLayout()
        timing_layout.setSpacing(2)                  # reduced from 3
        timing_layout.setContentsMargins(0, 0, 0, 0)

        duration_label = QLabel(f"↔ {duration_sec:.3f}s")
        duration_label.setStyleSheet(
            "color: #ffffaa; background: rgba(120, 120, 0, 0.3); "
            "border-radius: 4px; padding: 2px 6px; font-weight: bold;"  # reduced padding
        )
        duration_label.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 10))

        time_range = QLabel(f"{start_sec:.2f} → {end_sec:.2f}s")
        time_range.setStyleSheet("color: #bbbbbb; font-size: 9px;")
        time_range.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 9))

        timing_layout.addWidget(duration_label, alignment=Qt.AlignmentFlag.AlignCenter)
        timing_layout.addWidget(time_range, alignment=Qt.AlignmentFlag.AlignCenter)

        main_layout.addLayout(timing_layout)

        # Minimal container styling
        container.setStyleSheet("""
            QWidget {
                background: rgba(25, 30, 50, 0.45);
                border-radius: 6px;                  # reduced from 8px
            }
            QWidget:hover {
                background: rgba(35, 40, 65, 0.6);
            }
        """)

        self.content_layout.addWidget(container)

        QTimer.singleShot(0, lambda: self._scroll_to_bottom_smooth())

        # Removed summary update (line/char counts no longer displayed)
        pass

    def _scroll_to_bottom_smooth(self) -> None:
        """Smoothly scroll to the latest message."""
        scrollbar = self.scroll.verticalScrollBar()
        QApplication.processEvents()  # Ensure layout is updated
        QPropertyAnimation(
            scrollbar,
            b"value",
            self,
            startValue=scrollbar.value(),
            endValue=scrollbar.maximum(),
            duration=300,  # ms
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
        One-liner to get a perfectly centered, always-on-top, live subtitle overlay.
        Features (all automatic):
        - Creates QApplication if needed
        - Prevents quit on close/minimize
        - Centers on main screen
        - Graceful Ctrl+C shutdown
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
            # Position top-right (generic reusable logic)
            overlay.position_window_top_right(margin=30)
            overlay.logger.info("[green]Overlay positioned top-right[/]")

        QTimer.singleShot(50, _do_center)
        return overlay
