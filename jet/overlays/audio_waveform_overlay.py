import signal
import sys

import numpy as np
import pyqtgraph as pg
import sounddevice as sd
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# Configuration constants
OVERLAY_WIDTH = 450
OVERLAY_HEIGHT = 300  # Increased height for better visibility
DEFAULT_MARGIN_RIGHT = 20
DEFAULT_MARGIN_BOTTOM = 50
DEFAULT_MARGIN_TOP = 20

# Waveform constants
DEFAULT_SAMPLERATE = 44100  # Default audio sample rate (Hz)
DEFAULT_BLOCKSIZE = 512  # Default block size for audio chunks
DEFAULT_WINDOW_SIZE_SECONDS = 5.0
DEFAULT_AMPLITUDE_SCALE = 2.0  # Default amplitude multiplier

MIN_AMPLITUDE_SCALE = 0.1
MAX_AMPLITUDE_SCALE = 10.0
AMPLITUDE_SCALE_STEP = 0.1


class LiveAudioWaveform(QWidget):
    audio_data_ready = pyqtSignal(np.ndarray)

    def __init__(
        self,
        device=None,
        channels: int = 1,
        samplerate: int = DEFAULT_SAMPLERATE,
        blocksize: int = DEFAULT_BLOCKSIZE,
        window_size_seconds: float = DEFAULT_WINDOW_SIZE_SECONDS,
        amplitude_scale: float = DEFAULT_AMPLITUDE_SCALE,
        parent=None,
    ):
        super().__init__(parent)

        # Audio configuration
        self.channels = channels
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.buffer_size = int(window_size_seconds * samplerate)

        # Audio buffer
        self.audio_buffer = np.zeros((self.buffer_size, self.channels))
        self.write_index = 0

        # Amplitude scaling
        self.amplitude_scale = amplitude_scale
        self.auto_scale = False
        self.current_rms = 0.0
        self.update_counter = 0

        # Window configuration
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.resize(OVERLAY_WIDTH, OVERLAY_HEIGHT)

        # Position window
        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            self.move(
                geom.right() - self.width() - DEFAULT_MARGIN_RIGHT,
                geom.bottom() - self.height() - DEFAULT_MARGIN_BOTTOM,
            )

        # Build UI
        self._setup_ui()

        # Connect signals
        self.audio_data_ready.connect(self.update_plot)

        # Drag handling
        self.drag_position = None

        print(
            f"[INIT] LiveAudioWaveform initialized with amplitude_scale={amplitude_scale:.1f}x"
        )

    def _setup_ui(self):
        """Set up the user interface components."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Top bar with title and controls
        self._create_top_bar()

        # Controls bar for waveform height adjustment
        self._create_controls_bar()

        # Plot widget
        self._create_plot_widget()

        print("[UI] Setup complete")

    def _create_top_bar(self):
        """Create the title bar with close button."""
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

        # Add scale indicator
        self.scale_label = QLabel(f"Scale: {self.amplitude_scale:.1f}x")
        self.scale_label.setStyleSheet(
            "color: #00ffff; font-size: 11px; font-weight: bold;"
        )
        top_layout.addWidget(self.scale_label)

        top_layout.addSpacing(10)

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

    def _create_controls_bar(self):
        """Create controls for adjusting waveform height."""
        self.controls_bar = QWidget()
        self.controls_bar.setFixedHeight(40)
        self.controls_bar.setStyleSheet(
            "background-color: #333333; border-bottom: 1px solid #444;"
        )

        controls_layout = QHBoxLayout(self.controls_bar)
        controls_layout.setContentsMargins(8, 4, 8, 4)
        controls_layout.setSpacing(8)

        # Amplitude label
        amp_label = QLabel("Height:")
        amp_label.setStyleSheet("color: #cccccc; font-size: 11px; font-weight: bold;")
        controls_layout.addWidget(amp_label)

        # Amplitude slider (linear scale for better control)
        self.amp_slider = QSlider(Qt.Orientation.Horizontal)
        self.amp_slider.setRange(
            int(MIN_AMPLITUDE_SCALE * 100), int(MAX_AMPLITUDE_SCALE * 100)
        )
        self.amp_slider.setValue(int(self.amplitude_scale * 100))
        self.amp_slider.setStyleSheet(
            "QSlider::groove:horizontal { height: 6px; background: #555; border-radius: 3px; } "
            "QSlider::handle:horizontal { background: #00ffff; width: 16px; margin: -5px 0; "
            "border-radius: 8px; } "
            "QSlider::sub-page:horizontal { background: #00cccc; border-radius: 3px; }"
        )
        self.amp_slider.valueChanged.connect(self._on_amplitude_slider_changed)
        controls_layout.addWidget(self.amp_slider, 1)  # Stretch factor 1

        # Amplitude spinbox (shows percentage)
        self.amp_spinbox = QSpinBox()
        self.amp_spinbox.setRange(
            int(MIN_AMPLITUDE_SCALE * 100), int(MAX_AMPLITUDE_SCALE * 100)
        )
        self.amp_spinbox.setValue(int(self.amplitude_scale * 100))
        self.amp_spinbox.setSingleStep(10)  # 10% steps
        self.amp_spinbox.setSuffix("%")
        self.amp_spinbox.setFixedWidth(70)
        self.amp_spinbox.setStyleSheet(
            "QSpinBox { background: #444; color: #00ffff; border: 1px solid #555; "
            "border-radius: 3px; padding: 3px; font-weight: bold; }"
        )
        self.amp_spinbox.valueChanged.connect(self._on_amplitude_spinbox_changed)
        controls_layout.addWidget(self.amp_spinbox)

        # Auto-scale checkbox
        self.auto_scale_cb = QCheckBox("Auto")
        self.auto_scale_cb.setStyleSheet(
            "QCheckBox { color: #cccccc; font-size: 11px; font-weight: bold; spacing: 4px; } "
            "QCheckBox::indicator { width: 16px; height: 16px; } "
            "QCheckBox::indicator:unchecked { background: #555; border: 2px solid #666; "
            "border-radius: 3px; } "
            "QCheckBox::indicator:checked { background: #00ffff; border: 2px solid #00cccc; "
            "border-radius: 3px; }"
        )
        self.auto_scale_cb.toggled.connect(self._on_auto_scale_toggled)
        controls_layout.addWidget(self.auto_scale_cb)

        # RMS level indicator
        self.rms_label = QLabel("Level: -∞ dB")
        self.rms_label.setStyleSheet("color: #999999; font-size: 10px;")
        controls_layout.addWidget(self.rms_label)

        controls_layout.addStretch()

        self.main_layout.addWidget(self.controls_bar)
        print("[UI] Controls bar created")

    def _create_plot_widget(self):
        """Create the waveform plot widget with fixed Y-axis range."""
        self.plot_widget = pg.PlotWidget(title="")
        self.plot_widget.setBackground("#1e1e1e")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # CRITICAL FIX: Disable auto-ranging on Y-axis
        self.plot_widget.getPlotItem().vb.setMouseEnabled(x=True, y=False)

        # Disable auto-range for Y-axis
        self.plot_widget.getPlotItem().enableAutoRange(axis="y", enable=False)

        # Set initial Y range
        self._update_y_range()

        # Hide axes for cleaner look
        self.plot_widget.hideAxis("left")
        self.plot_widget.hideAxis("bottom")

        # Create curve with thicker line for better visibility
        self.curve = self.plot_widget.plot(pen=pg.mkPen("#00ffff", width=2.5))

        # Add horizontal reference lines at -1, 0, 1 for visual reference
        self._add_reference_lines()

        self.main_layout.addWidget(self.plot_widget, 1)  # Stretch factor 1
        print("[UI] Plot widget created with fixed Y-axis range")

    def _add_reference_lines(self):
        """Add horizontal reference lines for visual guidance."""
        # Center line (0)
        center_line = pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen("#ffffff", width=1, style=Qt.PenStyle.DashLine)
        )
        self.plot_widget.addItem(center_line)

        # Top and bottom lines at ±amplitude_scale
        self.top_line = pg.InfiniteLine(
            pos=self.amplitude_scale,
            angle=0,
            pen=pg.mkPen("#ff5555", width=1, style=Qt.PenStyle.DotLine),
        )
        self.bottom_line = pg.InfiniteLine(
            pos=-self.amplitude_scale,
            angle=0,
            pen=pg.mkPen("#ff5555", width=1, style=Qt.PenStyle.DotLine),
        )
        self.plot_widget.addItem(self.top_line)
        self.plot_widget.addItem(self.bottom_line)

        # Unity gain lines (±1.0)
        unity_top = pg.InfiniteLine(
            pos=1.0,
            angle=0,
            pen=pg.mkPen("#00ff00", width=1, style=Qt.PenStyle.DashLine),
        )
        unity_bottom = pg.InfiniteLine(
            pos=-1.0,
            angle=0,
            pen=pg.mkPen("#00ff00", width=1, style=Qt.PenStyle.DashLine),
        )
        self.plot_widget.addItem(unity_top)
        self.plot_widget.addItem(unity_bottom)

        print("[UI] Reference lines added")

    def _update_reference_lines(self):
        """Update reference line positions based on current scale."""
        if hasattr(self, "top_line"):
            self.top_line.setPos(self.amplitude_scale)
        if hasattr(self, "bottom_line"):
            self.bottom_line.setPos(-self.amplitude_scale)

    def _update_y_range(self):
        """Update the Y-axis range based on current amplitude scale."""
        # Add 10% headroom for visual comfort
        y_max = self.amplitude_scale * 1.1
        self.plot_widget.setYRange(-y_max, y_max)

        # CRITICAL: Disable auto-range again after setting range
        self.plot_widget.getPlotItem().enableAutoRange(axis="y", enable=False)

        # Update reference lines
        self._update_reference_lines()

        # Update scale label
        if hasattr(self, "scale_label"):
            self.scale_label.setText(f"Scale: {self.amplitude_scale:.1f}x")

        print(
            f"[RANGE] Y-axis range set to [-{y_max:.1f}, {y_max:.1f}] with scale {self.amplitude_scale:.1f}x"
        )

    def _on_amplitude_slider_changed(self, value):
        """Handle amplitude slider value change."""
        scale = value / 100.0
        if abs(scale - self.amplitude_scale) > 0.001:  # Avoid floating point issues
            self.amplitude_scale = scale
            self.amp_spinbox.blockSignals(True)
            self.amp_spinbox.setValue(int(scale * 100))
            self.amp_spinbox.blockSignals(False)
            self._update_y_range()
            print(
                f"[CONTROL] Amplitude scale changed to {scale:.2f}x ({value}%) via slider"
            )

    def _on_amplitude_spinbox_changed(self, value):
        """Handle amplitude spinbox value change."""
        scale = value / 100.0
        if abs(scale - self.amplitude_scale) > 0.001:
            self.amplitude_scale = scale
            self.amp_slider.blockSignals(True)
            self.amp_slider.setValue(int(scale * 100))
            self.amp_slider.blockSignals(False)
            self._update_y_range()
            print(
                f"[CONTROL] Amplitude scale changed to {scale:.2f}x ({value}%) via spinbox"
            )

    def _on_auto_scale_toggled(self, checked):
        """Handle auto-scale toggle."""
        self.auto_scale = checked
        if checked:
            self.amp_slider.setEnabled(False)
            self.amp_spinbox.setEnabled(False)
            print("[AUTO] Auto-scale enabled")
        else:
            self.amp_slider.setEnabled(True)
            self.amp_spinbox.setEnabled(True)
            # Reset to current slider value
            self.amplitude_scale = self.amp_slider.value() / 100.0
            self._update_y_range()
            print("[AUTO] Auto-scale disabled")

    def _calculate_rms(self, data):
        """Calculate RMS value of audio data."""
        if len(data) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(data))))

    def _calculate_db(self, rms):
        """Convert RMS to decibels."""
        if rms <= 0.000001:  # -120 dB threshold
            return float("-inf")
        return 20 * np.log10(rms)

    def _update_rms_display(self, rms):
        """Update the RMS level display with color coding."""
        db_level = self._calculate_db(rms)
        if db_level == float("-inf"):
            self.rms_label.setText("Level: -∞ dB")
            self.rms_label.setStyleSheet("color: #666666; font-size: 10px;")
        else:
            self.rms_label.setText(f"Level: {db_level:.1f} dB")
            # Color code based on level
            if db_level > -3:
                color = "#ff3333"  # Red - clipping danger
            elif db_level > -6:
                color = "#ff9900"  # Orange - hot
            elif db_level > -12:
                color = "#ffdd00"  # Yellow - good
            elif db_level > -20:
                color = "#00ff00"  # Green - normal
            elif db_level > -40:
                color = "#00ccff"  # Cyan - low
            else:
                color = "#6666ff"  # Blue - very low
            self.rms_label.setStyleSheet(
                f"color: {color}; font-size: 10px; font-weight: bold;"
            )

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for window dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Don't drag if clicking on controls
            if self.close_btn.geometry().contains(event.pos()):
                return
            if self.controls_bar.geometry().contains(event.pos()):
                return
            self.drag_position = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for window dragging."""
        if (
            event.buttons() == Qt.MouseButton.LeftButton
            and self.drag_position is not None
        ):
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        self.drag_position = None
        event.accept()

    def audio_callback(self, indata, frames, time, status):
        """Runs in a background thread. Must be fast and non-blocking."""
        if status:
            print(f"[AUDIO] Status: {status}", file=sys.stderr)
        self.audio_data_ready.emit(indata.copy())

    def update_plot(self, new_data):
        """Runs in the main GUI thread. Updates the circular buffer and plot."""
        self.update_counter += 1
        new_frames = new_data.shape[0]

        # Update circular buffer
        if self.write_index + new_frames <= self.buffer_size:
            self.audio_buffer[self.write_index : self.write_index + new_frames] = (
                new_data
            )
            self.write_index += new_frames
        else:
            remaining = self.buffer_size - self.write_index
            self.audio_buffer[self.write_index :] = new_data[:remaining]
            self.audio_buffer[: new_frames - remaining] = new_data[remaining:]
            self.write_index = (self.write_index + new_frames) % self.buffer_size

        # Get display data (first channel)
        display_data = np.roll(self.audio_buffer[:, 0], -self.write_index)

        # Calculate RMS for level display and auto-scaling
        self.current_rms = self._calculate_rms(display_data[-new_frames:])
        self._update_rms_display(self.current_rms)

        # Apply auto-scaling if enabled
        if self.auto_scale and self.current_rms > 0.0001:
            # Target RMS at 70% of current scale for good visibility
            target_scale = (self.amplitude_scale * 0.7) / max(self.current_rms, 0.0001)
            # Smooth the scale changes
            self.amplitude_scale = self.amplitude_scale * 0.95 + target_scale * 0.05
            # Clamp to reasonable range
            self.amplitude_scale = float(
                np.clip(self.amplitude_scale, MIN_AMPLITUDE_SCALE, MAX_AMPLITUDE_SCALE)
            )
            self._update_y_range()

        # Apply amplitude scaling
        scaled_data = display_data * self.amplitude_scale

        # Update the plot
        self.curve.setData(scaled_data)

        # Log periodically
        if self.update_counter % 50 == 0:
            print(
                f"[PLOT] Update #{self.update_counter}: "
                f"scale={self.amplitude_scale:.2f}x, "
                f"RMS={self.current_rms:.4f}, "
                f"peak={np.max(np.abs(scaled_data)):.3f}, "
                f"buffer={self.write_index}/{self.buffer_size}"
            )

    def start(self):
        """Start the audio stream."""
        try:
            self.stream = sd.InputStream(
                device=None,
                channels=self.channels,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                callback=self.audio_callback,
            )
            self.stream.start()
            print(
                f"[STREAM] Audio stream started: {self.channels}ch, "
                f"{self.samplerate}Hz, blocksize={self.blocksize}"
            )
        except sd.PortAudioError as e:
            print(f"[ERROR] Failed to start audio stream: {e}")
            print(
                "[ERROR] Please ensure a valid input device is connected and selected."
            )

    def stop(self):
        """Stop the audio stream."""
        if hasattr(self, "stream") and self.stream.active:
            self.stream.stop()
            self.stream.close()
            print("[STREAM] Audio stream stopped")

    def closeEvent(self, event):
        """Handle window close event."""
        print("[CLOSE] Closing LiveAudioWaveform...")
        self.stop()
        event.accept()


def setup_signal_handlers(app, window):
    """Setup graceful shutdown on Ctrl+C (SIGINT)."""

    def signal_handler(signum, frame):
        print("\n[SIGNAL] Shutting down gracefully...")
        window.stop()
        QTimer.singleShot(0, app.quit)

    signal.signal(signal.SIGINT, signal_handler)

    # Timer to keep Python processing signals
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(200)

    return signal_handler


if __name__ == "__main__":
    QApplication.setStyle("Fusion")
    app = QApplication(sys.argv)

    # Create window with enhanced amplitude
    window = LiveAudioWaveform(
        channels=1,
        samplerate=DEFAULT_SAMPLERATE,
        blocksize=DEFAULT_BLOCKSIZE,
        window_size_seconds=DEFAULT_WINDOW_SIZE_SECONDS,
        amplitude_scale=DEFAULT_AMPLITUDE_SCALE,
    )

    setup_signal_handlers(app, window)
    window.start()
    window.show()

    print("=" * 60)
    print("Live Audio Waveform started with FIXED height control")
    print("Default amplitude scale: 2.0x (200%)")
    print("✓ Y-axis auto-ranging DISABLED")
    print("✓ Manual height control via slider (10% - 1000%)")
    print("✓ Waveform fills entire height of the plot")
    print("✓ Reference lines show scale boundaries")
    print("=" * 60)

    sys.exit(app.exec())
