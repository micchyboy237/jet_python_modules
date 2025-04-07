import rumps
import psutil
import sys
import time
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton
from PyQt6.QtCore import Qt, QTimer


# --- RAM history buffer (stores (timestamp, used_memory_in_GB)) ---
ram_history = []


def add_ram_history_entry():
    mem = psutil.virtual_memory()
    ram_usage_gb = round(mem.used / (1024**3) * 2) / \
        2  # Round to nearest 0.5 GB
    ram_history.append((time.time(), ram_usage_gb))
    # Keep only entries from the last 3 hours
    three_hours_ago = time.time() - 3 * 60 * 60
    while ram_history and ram_history[0][0] < three_hours_ago:
        ram_history.pop(0)


class HistoryWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAM Usage History (Top 3 Peaks)")
        self.setGeometry(150, 150, 300, 150)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)

        self.layout = QVBoxLayout()
        self.history_label = QLabel("No history yet.", self)
        self.layout.addWidget(self.history_label)
        self.setLayout(self.layout)

    def update_history(self):
        if not ram_history:
            self.history_label.setText("No data available.")
            return
        top_peaks = sorted([entry[1]
                           for entry in ram_history], reverse=True)[:3]
        formatted = "\n".join(
            [f"Peak {i + 1}: {val:.1f} GB" for i, val in enumerate(top_peaks)])
        self.history_label.setText(formatted)


class RAMMonitorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAM Monitor")
        self.setGeometry(100, 100, 300, 150)  # Reduced size
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)

        layout = QVBoxLayout()
        self.ram_label = QLabel(self.get_ram_usage(), self)
        self.process_label = QLabel(self.get_top_processes(), self)

        self.refresh_button = QPushButton('Refresh', self)
        self.quit_button = QPushButton('Quit', self)

        self.refresh_button.clicked.connect(self.refresh)
        self.quit_button.clicked.connect(self.quit_app)

        layout.addWidget(self.ram_label)
        layout.addWidget(self.process_label)
        layout.addWidget(self.refresh_button)
        layout.addWidget(self.quit_button)

        self.setLayout(layout)

    def update_data(self):
        self.ram_label.setText(self.get_ram_usage())
        self.process_label.setText(self.get_top_processes())

    def get_ram_usage(self):
        mem = psutil.virtual_memory()
        return f'RAM Usage: {mem.used / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB ({mem.percent:.2f}%)'

    def get_top_processes(self):
        processes = []
        for p in psutil.process_iter(['name', 'memory_info']):
            try:
                if p.info['memory_info']:
                    processes.append(
                        (p.info['memory_info'].rss, p.info['name']))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        processes.sort(reverse=True, key=lambda x: x[0])
        top_processes = processes[:5]
        return "\n".join([f"{name}: {mem / (1024**3):.2f} GB" for mem, name in top_processes])

    def refresh(self):
        self.update_data()

    def quit_app(self):
        self.close()
        sys.exit()


class RAMMonitorApp(rumps.App):
    def __init__(self):
        super().__init__("RAM Monitor")
        self.set_icon(None)
        self.menu = ["Show History"]
        self.app = QApplication(sys.argv)

        # Create windows
        self.window = RAMMonitorWindow()
        self.history_window = HistoryWindow()

        # Get screen size for top-right positioning
        screen_geometry = self.app.primaryScreen().availableGeometry()
        margin = 20
        main_width, main_height = 300, 150

        # Position main window
        main_x = screen_geometry.right() - main_width - margin
        main_y = screen_geometry.top() + margin
        self.window.setGeometry(main_x, main_y, main_width, main_height)

        # Position history window directly below RAM window
        self.window.show()
        self.window.raise_()  # Ensure it's on top
        self.app.processEvents()  # Force geometry update before querying it

        # Get actual height in case size was adjusted by layout
        ram_window_geom = self.window.geometry()
        history_x = ram_window_geom.x()
        history_y = ram_window_geom.y() + ram_window_geom.height() + 10  # 10px gap
        history_width, history_height = 300, 150
        self.history_window.setGeometry(
            history_x, history_y, history_width, history_height)

        # Show history window
        self.history_window.update_history()
        self.history_window.show()

        # Set periodic RAM usage logging
        self.timer = QTimer()
        self.timer.timeout.connect(self.track_usage)
        self.timer.start(60000)

        self.update_ram_usage()

    def set_icon(self, path):
        self.icon = path  # Optional: add icon path here if needed

    def update_ram_usage(self):
        mem = psutil.virtual_memory()
        self.title = f'RAM: {mem.percent:.1f}%'
        self.window.ram_label.setText(self.window.get_ram_usage())

    def track_usage(self):
        add_ram_history_entry()
        self.update_ram_usage()

    @rumps.clicked("Show History")
    def toggle_history(self, _):
        if self.history_window.isVisible():
            self.history_window.hide()
        else:
            self.history_window.update_history()
            self.history_window.show()


def start_ram_monitor():
    app = RAMMonitorApp()
    app.run()


if __name__ == "__main__":
    start_ram_monitor()
