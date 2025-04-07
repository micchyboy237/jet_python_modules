import rumps
import psutil
import sys
import time
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit
)
from PyQt6.QtCore import Qt, QTimer

ram_history = []


def add_ram_history_entry():
    mem = psutil.virtual_memory()
    ram_usage_gb = round(mem.used / (1024**3) * 2) / 2
    ram_history.append((time.time(), ram_usage_gb))
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


class ProcessListWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("All Processes by Memory Usage")
        # Adjusted width for better layout
        self.setGeometry(200, 200, 500, 450)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)

        self.layout = QVBoxLayout()
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.layout.addWidget(self.text_area)

        # Pagination buttons
        button_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.page_label = QLabel()

        self.prev_button.clicked.connect(self.prev_page)
        self.next_button.clicked.connect(self.next_page)

        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.page_label)
        button_layout.addWidget(self.next_button)
        self.layout.addLayout(button_layout)

        self.setLayout(self.layout)

        self.page = 0
        self.items_per_page = 20  # Show 20 items per page
        self.processes = []

    def load_processes(self):
        processes = []
        for p in psutil.process_iter(['name', 'memory_info']):
            try:
                if p.info['memory_info']:
                    processes.append(
                        (p.info['memory_info'].rss, p.info['name']))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        self.processes = sorted(processes, reverse=True, key=lambda x: x[0])
        self.page = 0
        self.show_page()

    def show_page(self):
        start = self.page * self.items_per_page
        end = start + self.items_per_page
        page_items = self.processes[start:end]

        # Define maximum length for process names (e.g., 40 characters)
        max_name_length = 40
        formatted = "\n".join(
            [f"{i + 1 + start:<4} {name[:max_name_length]:<40} {mem / (1024**2):>10.2f} MB"
             for i, (mem, name) in enumerate(page_items)]
        )

        self.text_area.setText(formatted or "No processes to show")
        self.page_label.setText(
            f"Page {self.page + 1} / {max(1, (len(self.processes)-1)//self.items_per_page + 1)}")
        self.prev_button.setEnabled(self.page > 0)
        self.next_button.setEnabled(end < len(self.processes))

    def next_page(self):
        if (self.page + 1) * self.items_per_page < len(self.processes):
            self.page += 1
            self.show_page()

    def prev_page(self):
        if self.page > 0:
            self.page -= 1
            self.show_page()


class RAMMonitorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAM Monitor")
        self.setGeometry(100, 100, 300, 180)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)

        layout = QVBoxLayout()
        self.ram_label = QLabel(self.get_ram_usage(), self)
        self.process_label = QLabel(self.get_top_processes(), self)
        self.countdown_label = QLabel("Next update in: 10s", self)

        layout.addWidget(self.ram_label)
        layout.addWidget(self.process_label)
        layout.addWidget(self.countdown_label)

        self.refresh_button = QPushButton('Refresh', self)
        self.all_processes_button = QPushButton('Show All Processes', self)
        self.quit_button = QPushButton('Quit', self)

        self.refresh_button.clicked.connect(self.refresh)
        self.quit_button.clicked.connect(self.quit_app)

        layout.addWidget(self.all_processes_button)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.refresh_button)
        button_layout.addWidget(self.quit_button)
        layout.addLayout(button_layout)

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

    def update_countdown(self, seconds_left):
        self.countdown_label.setText(f"Next update in: {seconds_left}s")


class RAMMonitorApp(rumps.App):
    def __init__(self):
        super().__init__("RAM Monitor")
        self.set_icon(None)
        self.menu = ["Show History"]
        self.app = QApplication(sys.argv)

        self.window = RAMMonitorWindow()
        self.history_window = HistoryWindow()
        self.process_list_window = ProcessListWindow()

        self.window.all_processes_button.clicked.connect(
            self.show_all_processes)

        screen_geometry = self.app.primaryScreen().availableGeometry()
        margin = 20
        main_width, main_height = 300, 180
        main_x = screen_geometry.right() - main_width - margin
        main_y = screen_geometry.top() + margin
        self.window.setGeometry(main_x, main_y, main_width, main_height)

        # Position the process list window to the left of the RAM Monitor window by default
        process_list_x = main_x - 500 - margin  # Adjust 500 for window width
        process_list_y = main_y
        self.process_list_window.setGeometry(
            process_list_x, process_list_y, 500, 450)

        self.window.show()
        self.window.raise_()
        self.app.processEvents()

        ram_window_geom = self.window.geometry()
        history_x = ram_window_geom.x()
        history_y = ram_window_geom.y() + ram_window_geom.height() + 10
        self.history_window.setGeometry(history_x, history_y, 300, 150)

        self.history_window.update_history()
        self.history_window.show()

        self.timer = QTimer()
        self.timer.timeout.connect(self.track_usage)
        self.timer.start(10000)

        self.seconds_until_next = 10
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.start(1000)

        self.update_ram_usage()

        # New flag for process window visibility
        self.is_process_window_visible = False

        # Show by default
        self.show_all_processes()

    def set_icon(self, path):
        self.icon = path

    def update_ram_usage(self):
        mem = psutil.virtual_memory()
        self.title = f'RAM: {mem.percent:.1f}%'
        self.window.ram_label.setText(self.window.get_ram_usage())

    def track_usage(self):
        add_ram_history_entry()
        self.update_ram_usage()
        self.seconds_until_next = 10

    def update_countdown(self):
        self.seconds_until_next -= 1
        if self.seconds_until_next <= 0:
            self.seconds_until_next = 10
        self.window.update_countdown(self.seconds_until_next)

    def show_all_processes(self):
        if self.is_process_window_visible:
            self.process_list_window.hide()  # Hide the process window if it's currently visible
        else:
            self.process_list_window.load_processes()  # Load and show processes
            self.process_list_window.show()
        self.is_process_window_visible = not self.is_process_window_visible  # Toggle the flag

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
