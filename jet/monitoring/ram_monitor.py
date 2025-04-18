import rumps
import psutil
import sys
import time
import subprocess
import re
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit
)
from PyQt6.QtCore import Qt, QTimer

ram_history = []


def bytes_to_gb(bytes_value):
    """Convert bytes to GB with 2 decimal places."""
    return round(bytes_value / (1024 ** 3), 2)


def get_memory_stats():
    """Retrieve memory stats similar to Activity Monitor."""
    # Physical Memory
    physical_memory = int(subprocess.check_output(
        ["sysctl", "-n", "hw.memsize"]).decode().strip())
    physical_memory_gb = bytes_to_gb(physical_memory)

    # Memory Used (from top)
    top_output = subprocess.check_output(["top", "-l", "1"]).decode()
    memory_used_match = re.search(r"PhysMem:\s*(\d+\.?\d*[GM])", top_output)
    if memory_used_match:
        memory_used_str = memory_used_match.group(1)
        if "G" in memory_used_str:
            memory_used = float(memory_used_str.replace("G", ""))
        elif "M" in memory_used_str:
            memory_used = float(memory_used_str.replace("M", "")) / 1024
        memory_used_gb = bytes_to_gb(memory_used * (1024 ** 3))
    else:
        memory_used_gb = 0.0

    # Cached Files (from vm_stat)
    vm_stat_output = subprocess.check_output(["vm_stat"]).decode()
    page_size_match = re.search(r"page size of (\d+) bytes", vm_stat_output)
    cached_files_pages_match = re.search(
        r"Pages speculative:\s*(\d+)", vm_stat_output)
    if page_size_match and cached_files_pages_match:
        page_size = int(page_size_match.group(1))
        cached_files_pages = int(cached_files_pages_match.group(1))
        cached_files_bytes = cached_files_pages * page_size
        cached_files_gb = bytes_to_gb(cached_files_bytes)
    else:
        cached_files_gb = 0.0

    # Swap Used
    swap_output = subprocess.check_output(
        ["sysctl", "-n", "vm.swapusage"]).decode()
    swap_used_match = re.search(r"used = (\d+\.?\d*[GM])", swap_output)
    if swap_used_match:
        swap_used_str = swap_used_match.group(1)
        if "G" in swap_used_str:
            swap_used = float(swap_used_str.replace("G", ""))
        elif "M" in swap_used_str:
            swap_used = float(swap_used_str.replace("M", "")) / 1024
        swap_used_gb = bytes_to_gb(swap_used * (1024 ** 3))
    else:
        swap_used_gb = 0.0

    return {
        "physical_memory_gb": physical_memory_gb,
        "memory_used_gb": memory_used_gb,
        "cached_files_gb": cached_files_gb,
        "swap_used_gb": swap_used_gb
    }


def add_ram_history_entry():
    """Add a RAM usage entry to history based on Memory Used."""
    stats = get_memory_stats()
    ram_usage_gb = stats["memory_used_gb"]
    ram_history.append((time.time(), ram_usage_gb))
    three_hours_ago = time.time() - 3 * 60 * 60
    while ram_history and ram_history[0][0] < three_hours_ago:
        ram_history.pop(0)


class HistoryWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.top_n = 5
        self.setWindowTitle(f"RAM Usage History (Top {self.top_n} Peaks)")
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

        sorted_history = sorted(ram_history, key=lambda x: x[1], reverse=True)
        peaks = []
        prev_peak = None
        prev_time = None

        for timestamp, ram_usage in sorted_history:
            if not peaks or (ram_usage - prev_peak >= 0.5 or (timestamp - prev_time >= 10)):
                peaks.append((ram_usage, timestamp))
                prev_peak = ram_usage
                prev_time = timestamp
            if len(peaks) == self.top_n:
                break

        if not peaks:
            self.history_label.setText("No significant data.")
            return

        formatted = "\n".join([
            f"Peak {i + 1}: {ram_usage:.1f} GB (Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))})"
            for i, (ram_usage, timestamp) in enumerate(peaks)
        ])
        self.history_label.setText(formatted)


class ProcessListWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("All Processes by Memory Usage")
        self.setGeometry(200, 200, 500, 450)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)

        self.layout = QVBoxLayout()
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.layout.addWidget(self.text_area)

        self.refresh_button = QPushButton("Refresh", self)
        self.refresh_button.clicked.connect(self.load_processes)
        self.layout.addWidget(self.refresh_button)

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
        self.items_per_page = 20
        self.processes = []

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.load_processes)
        self.update_timer.start(10000)

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

        max_name_length = 40
        formatted = "\n".join(
            [f"{i + 1 + start:<4} {name[:max_name_length]:<40} {bytes_to_gb(mem):>10.2f} GB"
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
        stats = get_memory_stats()
        percent = (stats["memory_used_gb"] / stats["physical_memory_gb"]) * 100
        return (
            f"Physical Memory: {stats['physical_memory_gb']:.2f} GB\n"
            f"Memory Used: {stats['memory_used_gb']:.2f} GB ({percent:.1f}%)\n"
            f"Cached Files: {stats['cached_files_gb']:.2f} GB\n"
            f"Swap Used: {stats['swap_used_gb']:.2f} GB"
        )

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
        return "\n".join([f"{name}: {bytes_to_gb(mem):.2f} GB" for mem, name in top_processes])

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

        process_list_x = main_x - 500 - margin
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

        self.is_process_window_visible = False

    def set_icon(self, path):
        self.icon = path

    def update_ram_usage(self):
        stats = get_memory_stats()
        percent = (stats["memory_used_gb"] / stats["physical_memory_gb"]) * 100
        self.title = f'RAM: {percent:.1f}%'
        self.window.ram_label.setText(self.window.get_ram_usage())

    def track_usage(self):
        add_ram_history_entry()
        self.history_window.update_history()
        self.update_ram_usage()
        self.seconds_until_next = 10

    def update_countdown(self):
        self.seconds_until_next -= 1
        if self.seconds_until_next <= 0:
            self.seconds_until_next = 10
        self.window.update_countdown(self.seconds_until_next)

    def show_all_processes(self):
        if self.is_process_window_visible:
            self.process_list_window.hide()
        else:
            self.process_list_window.load_processes()
            self.process_list_window.show()
        self.is_process_window_visible = not self.is_process_window_visible

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
