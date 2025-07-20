import subprocess
from typing import NamedTuple, List, Optional, TypedDict
import psutil
import sys
import os
from dataclasses import dataclass


class SwapInfo(NamedTuple):
    """Represents swap memory information."""
    total: int
    used: int
    free: int


class ProcessInfo(TypedDict):
    """Typed dict for process information."""
    pid: int
    name: str
    memory_mb: float


@dataclass
class SwapManager:
    """Manages swap memory operations on macOS."""
    memory_threshold_mb: float = 100.0  # Configurable threshold for high-memory processes

    def get_swap_info(self) -> SwapInfo:
        """Retrieve swap memory information using sysctl."""
        try:
            result = subprocess.run(
                ["sysctl", "vm.swapusage"],
                capture_output=True,
                text=True,
                check=True
            )
            output = result.stdout
            # Parse output like: "vm.swapusage: total = 2048.00M  used = 512.00M  free = 1536.00M"
            parts = output.split()
            total = float(parts[4].rstrip("M"))
            used = float(parts[7].rstrip("M"))
            free = float(parts[10].rstrip("M"))
            return SwapInfo(total=total, used=used, free=free)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get swap info: {e}") from e

    def get_high_memory_processes(self) -> List[ProcessInfo]:
        """Return list of processes using memory above threshold."""
        processes: List[ProcessInfo] = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                memory_mb = proc.info['memory_info'].rss / \
                    1024 / 1024  # Convert bytes to MB
                if memory_mb > self.memory_threshold_mb:
                    processes.append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "memory_mb": memory_mb
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return sorted(processes, key=lambda x: x['memory_mb'], reverse=True)

    def terminate_process(self, pid: int) -> bool:
        """Terminate a process by PID."""
        try:
            proc = psutil.Process(pid)
            proc.terminate()
            proc.wait(timeout=3)  # Wait for termination
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as e:
            print(f"Failed to terminate process {pid}: {e}")
            return False

    def clear_system_cache(self) -> bool:
        """Clear system caches using sudo purge (requires admin privileges)."""
        try:
            subprocess.run(["sudo", "purge"], check=True,
                           capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to clear system cache: {e}")
            return False


def main() -> None:
    """Main function to check swap and manage high-memory processes."""
    manager = SwapManager(memory_threshold_mb=100.0)

    # Get swap info
    swap = manager.get_swap_info()
    print(
        f"Swap Usage: Total={swap.total:.2f}MB, Used={swap.used:.2f}MB, Free={swap.free:.2f}MB")

    # Check if swap usage is high
    if swap.used > swap.total * 0.5:  # Arbitrary threshold: 50% of total swap
        print("High swap usage detected. Listing high-memory processes...")
        processes = manager.get_high_memory_processes()
        if not processes:
            print("No processes found above memory threshold.")
            return

        for proc in processes:
            print(
                f"PID: {proc['pid']}, Name: {proc['name']}, Memory: {proc['memory_mb']:.2f}MB")

        # Prompt to terminate processes
        action = input(
            "Enter PID to terminate, 'all' to terminate all listed, 'cache' to clear system cache, or 'exit': ").strip()
        if action == "exit":
            sys.exit(0)
        elif action == "all":
            for proc in processes:
                if manager.terminate_process(proc['pid']):
                    print(f"Terminated process {proc['pid']} ({proc['name']})")
        elif action == "cache":
            if manager.clear_system_cache():
                print("System cache cleared successfully.")
        elif action.isdigit():
            pid = int(action)
            if any(proc['pid'] == pid for proc in processes):
                if manager.terminate_process(pid):
                    print(f"Terminated process {pid}")
            else:
                print(f"PID {pid} not in listed processes.")
        else:
            print("Invalid input.")
    else:
        print("Swap usage is within normal limits.")


if __name__ == "__main__":
    main()
