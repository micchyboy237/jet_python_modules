# subtitle_overlay.py
# One-liner: overlay = SubtitleOverlay.create()
# Then:      overlay.add_message("Your text here")   ← exactly what you want

import sys
import signal
import logging
from threading import Thread
import time
from typing import Optional, Awaitable
import asyncio
from concurrent.futures import Future

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea
)
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QObject, QTimer
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


class _Signals(QObject):
    _add_message = pyqtSignal(str)
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

        self.add_message("Live Subtitle Overlay • Ready")
        self.logger.info("[green]Ready – use .add_message('text')[/]")
        self._process_next_task()

    def _process_next_task(self) -> None:
        """Start the next queued async/sync task if available (runs sequentially)."""
        if not self._pending_tasks:
            return

        task, loading_widget = self._pending_tasks.pop(0)

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

        try:
            message, loading_widget = future.result()
        except Exception as e:
            message = f"[Error] {e}"
            # loading_widget = getattr(future, "_loading_widget", None)

        if not loading_widget or not loading_widget.parent():
            return

        # Stop the spinner animation immediately before deleting the widget
        timer = loading_widget.property("_spinner_timer")
        if timer and isinstance(timer, QTimer) and timer.isActive():
            timer.stop()

        idx = self.content_layout.indexOf(loading_widget)
        if idx == -1:
            return

        self.content_layout.takeAt(idx)
        loading_widget.deleteLater()

        new_label = QLabel(message)
        new_label.setWordWrap(True)
        new_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        new_label.setStyleSheet("color: white; padding: 4px;")
        new_label.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 24))
        self.content_layout.insertWidget(idx, new_label)

        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()))

        self.history.append(message)
        total_chars = sum(len(line) for line in self.history)
        self.summary.setText(f"{len(self.history)} lines • {total_chars:,} chars")

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
    def add_message(self, text: str):
        """Add a message (subtitled line) from anywhere, any thread."""
        if not text or not str(text).strip():
            return
        self.signals._add_message.emit(str(text).strip())

    def add_task(self, func: callable, *args, **kwargs) -> None:
        """
        Add a sync/async callable for sequential execution.
        Its return/awaited value is displayed as a message.
        Displays a "Processing" row until done.
        """
        async def _wrapper() -> str:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return str(result)

        # Create a loading widget (spinner + text)
        loading_widget = QWidget()
        loading_layout = QHBoxLayout(loading_widget)
        loading_layout.setContentsMargins(10, 8, 10, 8)
        loading_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        spinner = QLabel()
        self._setup_spinner_animation(spinner, loading_widget)

        loading_text = QLabel("Processing")
        loading_text.setStyleSheet("color: #ffff66; font-style: italic;")
        loading_text.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 24))

        loading_layout.addStretch()
        loading_layout.addWidget(spinner)
        loading_layout.addWidget(loading_text)
        loading_layout.addStretch()

        self.content_layout.addWidget(loading_widget)
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()))
        self.history.append("Processing…")
        total_chars = sum(len(line) for line in self.history)
        self.summary.setText(f"{len(self.history)} lines • {total_chars:,} chars")

        self._pending_tasks.append((_wrapper(), loading_widget))
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
    def _do_add_message(self, text: str):
        self.history.append(text)

        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: white; padding: 4px;")
        lbl.setFont(QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 24))

        self.content_layout.addWidget(lbl)

        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()))

        total_chars = sum(len(line) for line in self.history)
        self.summary.setText(f"{len(self.history)} lines • {total_chars:,} chars")

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


if __name__ == "__main__":
    # demo_threading()
    demo_async()
