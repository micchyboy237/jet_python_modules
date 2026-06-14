import signal
import sys

import numpy as np
import pyqtgraph as pg
import sounddevice as sd
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

OVERLAY_WIDTH = 450
OVERLAY_HEIGHT = 200
DEFAULT_MARGIN_RIGHT = 20
DEFAULT_MARGIN_BOTTOM = 20
DEFAULT_MARGIN_TOP = 20  # Kept for potential future use


class LiveAudioWaveform(QWidget):
    # Signal to safely pass audio data from the sounddevice thread to the GUI thread
    audio_data_ready = pyqtSignal(np.ndarray)

    def __init__(
        self,
        device=None,
        channels: int = 1,
        samplerate: int = 44100,
        blocksize: int = 512,
        window_size_seconds: float = 1.5,
        parent=None,
    ):
        super().__init__(parent)

        self.channels = channels
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.buffer_size = int(window_size_seconds * samplerate)

        # Circular buffer for seamless scrolling
        self.audio_buffer = np.zeros((self.buffer_size, self.channels))
        self.write_index = 0

        # --- UI Setup for Floating Panel ---
        # Frameless + Always on Top + Tool window (hides from taskbar on some OS)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.resize(OVERLAY_WIDTH, OVERLAY_HEIGHT)
        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            # Bottom-right positioning with configurable margins
            self.move(
                geom.right() - self.width() - DEFAULT_MARGIN_RIGHT,
                geom.bottom() - self.height() - DEFAULT_MARGIN_BOTTOM,
            )

        # Main Layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Top Drag Bar
        self.top_bar = QWidget()
        self.top_bar.setFixedHeight(28)
        self.top_bar.setStyleSheet(
            "background-color: #2b2b2b; border-bottom: 1px solid #444;"
        )
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(8, 0, 8, 0)

        self.title_label = QLabel("🎙️ Live Audio Waveform")
        self.title_label.setStyleSheet(
            "color: #e0e0e0; font-size: 13px; font-weight: bold;"
        )
        top_layout.addWidget(self.title_label)

        top_layout.addStretch()

        self.close_btn = QPushButton("×")
        self.close_btn.setFixedSize(24, 24)
        self.close_btn.setStyleSheet(
            "QPushButton { background-color: #ff5555; color: white; border: none; "
            "font-weight: bold; font-size: 16px; border-radius: 12px; } "
            "QPushButton:hover { background-color: #ff3333; }"
        )
        self.close_btn.clicked.connect(self.close)
        top_layout.addWidget(self.close_btn)

        self.main_layout.addWidget(self.top_bar)

        # Pyqtgraph Plot Widget
        self.plot_widget = pg.PlotWidget(title="")
        self.plot_widget.setBackground("#1e1e1e")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setYRange(-1.0, 1.0)
        self.plot_widget.setMouseEnabled(x=True, y=False)  # Allow horizontal zoom/pan
        self.plot_widget.hideAxis("left")  # Hide Y-axis numbers for a cleaner look
        self.plot_widget.hideAxis("bottom")

        self.main_layout.addWidget(self.plot_widget)

        # Waveform Curve (Cyan color, 1.5px width)
        self.curve = self.plot_widget.plot(pen=pg.mkPen("#00ffff", width=1.5))

        # Connect thread-safe signal
        self.audio_data_ready.connect(self.update_plot)

        # Dragging state
        self.drag_position = None

    # --- Mouse Events for Dragging the Frameless Window ---
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            # Prevent dragging if clicking the close button
            if self.close_btn.geometry().contains(event.pos()):
                return
            self.drag_position = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if (
            event.buttons() == Qt.MouseButton.LeftButton
            and self.drag_position is not None
        ):
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.drag_position = None
        event.accept()

    # --- Audio Processing ---
    def audio_callback(self, indata, frames, time, status):
        """Runs in a background thread. Must be fast and non-blocking."""
        if status:
            print(f"Audio Status: {status}", file=sys.stderr)

        # Emit a copy of the data to the GUI thread
        self.audio_data_ready.emit(indata.copy())

    def update_plot(self, new_data):
        """Runs in the main GUI thread. Updates the circular buffer and plot."""
        new_frames = new_data.shape[0]

        # Write new data into the circular buffer
        if self.write_index + new_frames <= self.buffer_size:
            self.audio_buffer[self.write_index : self.write_index + new_frames] = (
                new_data
            )
            self.write_index += new_frames
        else:
            # Handle wrap-around
            remaining = self.buffer_size - self.write_index
            self.audio_buffer[self.write_index :] = new_data[:remaining]
            self.audio_buffer[: new_frames - remaining] = new_data[remaining:]
            self.write_index = (self.write_index + new_frames) % self.buffer_size

        # Roll the buffer for display so the newest data is always on the right,
        # preventing a visual "jump" line at the wrap-around point.
        # We plot only the first channel (index 0) to keep it clean.
        display_data = np.roll(self.audio_buffer[:, 0], -self.write_index)
        self.curve.setData(display_data)

    # --- Lifecycle Management ---
    def start(self):
        try:
            self.stream = sd.InputStream(
                device=None,  # Defaults to system default input
                channels=self.channels,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                callback=self.audio_callback,
            )
            self.stream.start()
        except sd.PortAudioError as e:
            print(f"Failed to start audio stream: {e}")
            print("Please ensure a valid input device is connected and selected.")

    def stop(self):
        if hasattr(self, "stream") and self.stream.active:
            self.stream.stop()
            self.stream.close()

    def closeEvent(self, event):
        self.stop()
        event.accept()


def setup_signal_handlers(app, window):
    """Setup graceful shutdown on Ctrl+C (SIGINT)."""

    def signal_handler(signum, frame):
        print("\nShutting down gracefully...")
        window.stop()
        QTimer.singleShot(0, app.quit)

    # Handle SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    # Use a timer to periodically process signals in the Qt event loop
    timer = QTimer()
    timer.timeout.connect(lambda: None)  # Python signal processing works here
    timer.start(200)  # Check every 200ms

    return signal_handler


if __name__ == "__main__":
    # Optional: Set a modern application style
    QApplication.setStyle("Fusion")

    app = QApplication(sys.argv)

    # Initialize the reusable widget
    # Adjust 'channels' to 2 if you want to capture stereo (modify update_plot to handle it)
    window = LiveAudioWaveform(
        channels=1,
        samplerate=44100,
        blocksize=512,  # Lower = lower latency, higher CPU. 512 is a good balance.
        window_size_seconds=1.5,  # How many seconds of audio to show on screen
    )

    # Setup graceful shutdown on Ctrl+C
    setup_signal_handlers(app, window)

    window.start()
    window.show()

    sys.exit(app.exec())
