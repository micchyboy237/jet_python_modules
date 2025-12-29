# subtitle_overlay.py
# One-liner: overlay = SubtitleOverlay.create()
# Then:      overlay.add_message("Your text here")   ← exactly what you want

import sys
import signal
import logging
from threading import Thread
import time
from typing import Optional, Awaitable, TypedDict
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
    logger = logging.getLogger("SubtitleOverlay")
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


class SubtitleOverlay(QWidget):
    """
    Modern, thread-safe, live overlay for displaying subtitles, transcripts, or status messages.

    Usage:
        overlay = SubtitleOverlay.create()
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

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 190); border-radius: 16px;")
        self.setFixedSize(960, 600)

        self._drag_pos = QPoint()
        self._build_ui()
        self._connect_signals()

        # ------------------------------------------------------------------
        # ASYNCIO/TASK INTEGRATION (Always sequential)
        # ------------------------------------------------------------------
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_tasks: list[tuple[Awaitable[str], QWidget]] = []

        self.add_message(
            translated_text="Live Subtitle Overlay • Ready",
            source_text="",
        )
        self.logger.info("[green]Ready – use .add_message('text')[/]")
        self._process_next_task()

    def _process_next_task(self) -> None:
        """Start the next queued async/sync task if available (runs sequentially)."""
        if not self._pending_tasks:
            return

        task, loading_layout, loading_widget = self._pending_tasks[0]  # Keep in queue until done

        # --- Replace "Pending" with spinner + "Processing" ---
        loading_layout.takeAt(0)  # clear existing widgets/stretches
        while loading_layout.count():
            item = loading_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        spinner = QLabel()
        self._setup_spinner_animation(spinner, loading_widget)

        processing_text = QLabel("Processing")
        processing_text.setStyleSheet("color: #ffff66; font-style: italic; background-color: rgba(180, 140, 0, 0.2); border-radius: 8px; padding: 4px 12px;")
        processing_text.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 24))

        loading_layout.addStretch()
        loading_layout.addWidget(spinner)
        loading_layout.addWidget(processing_text)
        loading_layout.addStretch()

        def run_in_thread() -> tuple[str, QWidget]:
            try:
                result = asyncio.run(task)
                return str(result), loading_widget
            except Exception as e:
                return f"[Error] {e}", loading_widget

        future = self._executor.submit(run_in_thread)
        QTimer.singleShot(100, lambda: self._check_future_done(future))

    def _check_future_done(self, future: Future) -> None:
        if future.done():
            self._on_task_finished(future)
        else:
            QTimer.singleShot(100, lambda: self._check_future_done(future))

    def _on_task_finished(self, future: Future) -> None:
        """Handles completion of an add_task call."""
        if future.cancelled():
            return

        # Remove the finished task from pending queue
        if self._pending_tasks:
            self._pending_tasks.pop(0)

        try:
            message, loading_widget = future.result()
        except Exception as e:
            message = f"[Error] {e}"

        if not loading_widget or not loading_widget.parent():
            if self._pending_tasks:
                self._process_next_task()
            return

        # Stop spinner
        timer = loading_widget.property("_spinner_timer")
        if timer and isinstance(timer, QTimer) and timer.isActive():
            timer.stop()

        idx = self.content_layout.indexOf(loading_widget)
        if idx == -1:
            if self._pending_tasks:
                self._process_next_task()
            return

        self.content_layout.takeAt(idx)
        loading_widget.deleteLater()

        # Replace with final message
        new_label = QLabel(message)
        new_label.setWordWrap(True)
        new_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        new_label.setStyleSheet("color: white; padding: 4px;")
        new_label.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 24))
        self.content_layout.insertWidget(idx, new_label)

        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()))

        # Update history: replace the "Pending" entry
        pending_idx = next((i for i, h in enumerate(self.history) if h == "Pending"), None)
        if pending_idx is not None:
            self.history[pending_idx] = message
        else:
            self.history.append(message)

        total_chars = sum(len(line) for line in self.history)
        self.summary.setText(f"{len(self.history)} lines • {total_chars:,} chars")

        # Continue with next task
        if self._pending_tasks:
            self._process_next_task()

    def closeEvent(self, event):
        self._executor.shutdown(wait=False)
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

        # --- Content area (scrollable + summary) ---
        self.content_area = QVBoxLayout()
        self.content_area.setSpacing(8)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.content_layout.setSpacing(10)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.scroll.setWidget(self.content)

        self.summary = QLabel("0 lines • 0 chars")
        self.summary.setStyleSheet("""
            color: #aaffaa; font-size: 15px; padding: 8px;
            background: rgba(0, 100, 0, 0.5); border-radius: 8px;
        """)
        self.summary.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.content_area.addWidget(self.scroll, stretch=1)
        self.content_area.addWidget(self.summary)

        main.addLayout(self.content_area, stretch=1)

        self.min_btn.clicked.connect(self.toggle_minimize)
        self._is_minimized = False
        self._original_size = None

    def _connect_signals(self):
        self.signals._add_message.connect(self._do_add_message)
        self.signals._clear.connect(self.clear)
        self.signals._toggle_minimize.connect(self.toggle_minimize)

    def _center_window(self):
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.center().x() - self.width() // 2,
                  screen.center().y() - self.height() // 2)

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

    def add_task(self, func: callable, *args, **kwargs) -> None:
        """
        Add a sync/async callable for sequential execution.
        Its return/awaited value is displayed as a message.
        Displays a "Pending" row until processing starts, then replaces it with spinner + "Processing".
        """
        async def _wrapper() -> str:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return str(result)

        # Create initial "Pending" loading widget
        loading_widget = QWidget()
        loading_layout = QHBoxLayout(loading_widget)
        loading_layout.setContentsMargins(10, 8, 10, 8)
        loading_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        pending_text = QLabel("Pending")
        pending_text.setStyleSheet("color: #88ccff; font-style: italic; background-color: rgba(0, 80, 160, 0.2); border-radius: 8px; padding: 4px 12px;")
        pending_text.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 24))

        loading_layout.addStretch()
        loading_layout.addWidget(pending_text)
        loading_layout.addStretch()

        self.content_layout.addWidget(loading_widget)
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()))
        self.history.append("Pending")
        total_chars = sum(len(line) for line in self.history)
        self.summary.setText(f"{len(self.history)} lines • {total_chars:,} chars")

        # Store layout and widget for later replacement when processing starts
        self._pending_tasks.append((_wrapper(), loading_layout, loading_widget))
        if len(self._pending_tasks) == 1:  # Only process if this is becoming the active task
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

        timer = QTimer(self)  # parented to the overlay for longevity
        timer.timeout.connect(update_frame)
        timer.start(80)

        # Store the timer on the loading row so we can stop it on completion
        parent_widget.setProperty("_spinner_timer", timer)

        # Auto-stop the timer when the loading row is destroyed
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

        # Bilingual container widget
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(8)
        container_layout.setContentsMargins(20, 16, 20, 16)

        # Translated text – large, bold, prominent
        trans_label = QLabel(translated_text)
        trans_label.setWordWrap(True)
        trans_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        trans_label.setStyleSheet(
            "color: #ffffff; background: rgba(255, 255, 255, 0.12); "
            "border-radius: 12px; padding: 16px;"
        )
        trans_label.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 34, QFont.Weight.Bold))
        container_layout.addWidget(trans_label)

        # Source text – smaller, italic, only shown if present
        if source_text:
            src_label = QLabel(source_text)
            src_label.setWordWrap(True)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet(
                "color: #bbddff; font-style: italic; background: rgba(100, 140, 255, 0.18); "
                "border-radius: 10px; padding: 10px; margin-top: 4px;"
            )
            src_label.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 22))
            container_layout.addWidget(src_label)

        # Timing metadata row
        timing_layout = QHBoxLayout()
        timing_layout.setSpacing(20)

        start_label = QLabel(f"{start_sec:.3f}s")
        start_label.setStyleSheet("color: #88ff88; background: rgba(0, 100, 0, 0.3); border-radius: 6px; padding: 4px 10px;")
        start_label.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 14))

        end_label = QLabel(f"{end_sec:.3f}s")
        end_label.setStyleSheet("color: #ff8888; background: rgba(100, 0, 0, 0.3); border-radius: 6px; padding: 4px 10px;")
        end_label.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 14))

        duration_label = QLabel(f"↔ {duration_sec:.3f}s")
        duration_label.setStyleSheet("color: #ffff88; background: rgba(100, 100, 0, 0.3); border-radius: 6px; padding: 4px 10px;")
        duration_label.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 14, QFont.Weight.Bold))

        timing_layout.addStretch()
        timing_layout.addWidget(start_label)
        timing_layout.addWidget(duration_label)
        timing_layout.addWidget(end_label)
        timing_layout.addStretch()

        timing_container = QWidget()
        timing_container.setLayout(timing_layout)
        timing_container.setStyleSheet("background: rgba(40, 40, 60, 0.4); border-radius: 8px; margin-top: 8px;")
        
        container_layout.addWidget(timing_container)

        # Overall entry background
        container.setStyleSheet("""
            QWidget {
                background: rgba(25, 25, 45, 0.55);
                border-radius: 16px;
                border: 1px solid rgba(100, 100, 150, 0.15);
            }
        """)

        self.content_layout.addWidget(container)

        # Auto-scroll to the bottom with smooth animation
        QTimer.singleShot(0, lambda: self._scroll_to_bottom_smooth())

        total_chars = sum(len(line) for line in self.history)
        self.summary.setText(f"{len(self.history)} lines • {total_chars:,} chars")

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
    def create(cls, app: Optional[QApplication] = None, title: Optional[str] = None) -> 'SubtitleOverlay':
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
            screen = app.primaryScreen().availableGeometry()
            overlay.move(
                screen.center().x() - overlay.width() // 2,
                screen.center().y() - overlay.height() // 2,
            )
            overlay.logger.info("[green]Overlay centered perfectly[/]")

        QTimer.singleShot(50, _do_center)
        return overlay


# Async tasks demo
def demo_async():
    async def long_running_translation(text: str) -> str:
        await asyncio.sleep(1.0)
        return f"Translated: {text.upper()}"

    overlay = SubtitleOverlay.create(title="Async Demo")
    overlay.add_task(long_running_translation, "hello world")
    overlay.add_task(long_running_translation, "first")
    overlay.add_task(long_running_translation, "second")
    overlay.add_task(long_running_translation, "third")
    sys.exit(QApplication.exec())

# Threading demo
def demo_threading():
    overlay = SubtitleOverlay.create(title="Threading Demo")

    def demo():
        msgs = [
            "overlay.add_message() works!",
            "Super clean API",
            "Thread-safe",
            "Perfect for live translation",
            "Works on Windows + macOS",
            "This is a very long message to test word wrapping and auto-scrolling behavior during real-time streaming...",
        ]
        for i, msg in enumerate(msgs * 20):
            time.sleep(0.5)
            overlay.add_message(msg)
            if i == 10:
                overlay.toggle_minimize()
                time.sleep(0.5)
                overlay.toggle_minimize()

    Thread(target=demo, daemon=True).start()
    sys.exit(QApplication.exec())


# Fixed: subtitle_overlay.py – only the demo_subtitle_metadata() function

def demo_subtitle_metadata() -> None:
    """
    Demonstrates the new metadata-rich add_message API with realistic live subtitle timing.
    Shows English translation on screen while preserving start/end times, duration, and original Japanese (source) text.
    """
    overlay = SubtitleOverlay.create(title="Live Subtitles – Metadata Demo")

    # Simulated real-world segments – correct keys
    demo_segments = [
        {
            "translated_text": "Hello, how are you today?",
            "source_text": "こんにちは、今日はお元気ですか？",
            "start_sec": 1.250,
            "end_sec": 4.180,
        },
        {
            "translated_text": "I'm doing great, thank you!",
            "source_text": "元気です、ありがとう！",
            "start_sec": 4.500,
            "end_sec": 6.920,
        },
        {
            "translated_text": "That's wonderful to hear.",
            "source_text": "それは素晴らしいですね。",
            "start_sec": 7.100,
            "end_sec": 9.450,
        },
        {
            "translated_text": "What are your plans for the weekend?",
            "source_text": "週末の予定は何ですか？",
            "start_sec": 9.800,
            "end_sec": 12.300,
        },
    ]

    def feed_subtitles() -> None:
        import time

        # Start from the first real segment (skip the initial "Ready" message)
        for idx, seg in enumerate(demo_segments, start=2):
            # Natural pacing based on actual segment duration
            time.sleep(seg["end_sec"] - seg["start_sec"] + 0.4)

            overlay.add_message(
                translated_text=seg["translated_text"],
                start_sec=seg["start_sec"],
                end_sec=seg["end_sec"],
                duration_sec=round(seg["end_sec"] - seg["start_sec"], 3),
                source_text=seg["source_text"],
            )

            # Print correct metadata table (now shows proper values)
            latest = overlay.message_history[-1]
            from rich.table import Table
            from rich import print as rprint

            table = Table(title=f"Subtitle {idx} metadata")
            table.add_column("Field")
            table.add_column("Value")
            for k, v in latest.items():
                table.add_row(k, str(v))
            rprint(table)

    from threading import Thread
    Thread(target=feed_subtitles, daemon=True).start()

    sys.exit(QApplication.exec())
    

if __name__ == "__main__":
    # Uncomment one of the demos to run
    # demo_async()
    # demo_threading()
    demo_subtitle_metadata()
