import sys
import psutil
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt6.QtCore import QTimer


def get_ram_usage():
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)
    used_gb = mem.used / (1024 ** 3)
    return f'RAM Usage: {used_gb:.2f} GB / {total_gb:.2f} GB ({mem.percent:.2f}%)'


def get_top_processes():
    processes = []
    for p in psutil.process_iter(['name', 'memory_info']):
        try:
            if p.info['memory_info']:
                processes.append((p.info['memory_info'].rss, p.info['name']))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    processes.sort(reverse=True, key=lambda x: x[0])
    top_processes = processes[:5]
    return '\n'.join([f'{name}: {mem / (1024 ** 3):.2f} GB' for mem, name in top_processes])


class RAMMonitor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Real-Time RAM Monitor')
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()
        self.ram_label = QLabel(get_ram_usage(), self)
        self.process_label = QLabel(get_top_processes(), self)
        layout.addWidget(self.ram_label)
        layout.addWidget(self.process_label)
        self.setLayout(layout)

        # Timer to update RAM usage
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ram_usage)
        self.timer.start(1000)  # Update every 1 second

    def update_ram_usage(self):
        self.ram_label.setText(get_ram_usage())
        self.process_label.setText(get_top_processes())


def start_ram_monitor():
    app = QApplication(sys.argv)
    window = RAMMonitor()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    start_ram_monitor()
