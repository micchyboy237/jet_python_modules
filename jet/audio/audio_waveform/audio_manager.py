import queue
import threading
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd

# Type alias for clarity: Receive a numpy array of samples
AudioCallback = Callable[[np.ndarray], None]


class AudioStreamManager:
    """
    A reusable engine for capturing live audio and distributing
    samples to registered observers/callbacks.
    """

    def __init__(
        self, samplerate: int = 16000, block_size: int = 512, max_queue_size: int = 400
    ) -> None:
        self.samplerate = samplerate
        self.block_size = block_size

        # Threading and Data Flow
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue_size)
        self._observers: List[AudioCallback] = []
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Audio Stream
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.samplerate,
            blocksize=self.block_size,
            callback=self._sd_callback,
        )

    def add_observer(self, callback: AudioCallback) -> None:
        """Register a new handler for live audio samples."""
        if callback not in self._observers:
            self._observers.append(callback)

    def remove_observer(self, callback: AudioCallback) -> None:
        """Unsubscribe a handler."""
        if callback in self._observers:
            self._observers.remove(callback)

    def _sd_callback(self, indata, frames, time, status) -> None:
        """Internal sounddevice callback to move data to the queue."""
        if status:
            print(f"Stream Status: {status}")

        # Ensure we are working with a copy to prevent memory corruption
        samples = indata[:, 0].astype(np.float32).copy()

        try:
            # Drop oldest frame if queue is full to prevent latency buildup
            if self._audio_queue.full():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    pass
            self._audio_queue.put_nowait(samples)
        except queue.Full:
            pass

    def _processing_loop(self) -> None:
        """Background thread that consumes queue and notifies observers."""
        while self._running:
            try:
                # Blocks for a short time to keep thread alive but responsive to shutdown
                samples = self._audio_queue.get(timeout=0.1)

                # Notify all observers
                for callback in self._observers:
                    try:
                        callback(samples)
                    except Exception as e:
                        print(f"Error in observer {callback.__name__}: {e}")

            except queue.Empty:
                continue

    def start(self) -> None:
        """Start the audio stream and the background processing thread."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self._worker_thread.start()
        self.stream.start()
        print(f"Audio stream started at {self.samplerate}Hz...")

    def stop(self) -> None:
        """Gracefully stop the stream and background thread."""
        self._running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)
        print("Audio stream stopped.")


# --- Example Usage ---
if __name__ == "__main__":
    import time

    # 1. Define your custom logic (VAD, Plotting, Logging, etc.)
    def volume_monitor(samples: np.ndarray):
        rms = np.sqrt(np.mean(samples**2))
        if rms > 0.05:
            print(f"Peak detected! RMS: {rms:.4f}")

    def data_logger(samples: np.ndarray):
        # Pretend we're saving to a file or database
        pass

    # 2. Initialize and Run
    manager = AudioStreamManager(samplerate=16000)
    manager.add_observer(volume_monitor)
    manager.add_observer(data_logger)

    try:
        manager.start()
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop()
