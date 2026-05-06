import queue
import threading
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd

# Type alias for clarity
AudioCallback = Callable[[np.ndarray], None]


class AudioStreamManager:
    """
    Reusable engine for capturing live audio, optimized for FireRedVAD.

    - samplerate = 16000 Hz (required)
    - block_size = multiple of 160 (recommended)
    - dtype = int16 (preferred for VAD)
    """

    def __init__(
        self,
        samplerate: int = 16000,
        max_queue_size: int = 400,
        block_size: int = 480,
        dtype: str | None = "float32",
        # block_size: int = 160,  # Changed default
        # dtype: str = "int16",  # Added explicit dtype
    ) -> None:
        self.samplerate = samplerate
        self.block_size = block_size
        self.dtype = dtype

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
            dtype=self.dtype,  # Explicit int16
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
        """Internal sounddevice callback."""
        if status:
            print(f"Stream Status: {status}")

        # indata shape: (block_size, channels) -> take mono channel
        samples = indata[:, 0].copy()  # int16 copy

        try:
            if self._audio_queue.full():
                try:
                    self._audio_queue.get_nowait()  # drop oldest
                except queue.Empty:
                    pass
            self._audio_queue.put_nowait(samples)
        except queue.Full:
            pass

    def _processing_loop(self) -> None:
        """Background thread that notifies observers."""
        while self._running:
            try:
                samples = self._audio_queue.get(timeout=0.1)

                for callback in self._observers:
                    try:
                        callback(samples)
                    except Exception as e:
                        print(
                            f"Error in observer {getattr(callback, '__name__', callback)}: {e}"
                        )

            except queue.Empty:
                continue

    def start(self) -> None:
        """Start the audio stream and processing thread."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self._worker_thread.start()
        self.stream.start()
        print(
            f"Audio stream started at {self.samplerate}Hz (block_size={self.block_size}, dtype={self.dtype})"
        )

    def stop(self) -> None:
        """Gracefully stop everything."""
        self._running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        print("Audio stream stopped.")


# ============================
# Example Usage with FireRedVAD
# ============================
if __name__ == "__main__":
    import time

    from fireredvad import FireRedStreamVad, FireRedStreamVadConfig

    # --- VAD Observer ---
    class FireRedVADObserver:
        def __init__(self):
            vad_config = FireRedStreamVadConfig(
                use_gpu=False,
                smooth_window_size=5,
                speech_threshold=0.4,
                pad_start_frame=5,
                min_speech_frame=8,
                max_speech_frame=2000,
                min_silence_frame=20,
                chunk_max_frame=30000,
            )
            self.vad = FireRedStreamVad.from_pretrained(
                "pretrained_models/FireRedVAD/Stream-VAD",  # or HF path
                vad_config,
            )
            self.buffer = np.array([], dtype=np.int16)

        def __call__(self, samples: np.ndarray):
            """Called for every incoming audio block (160 samples)."""
            self.buffer = np.concatenate((self.buffer, samples))

            # Process in multiples of 160 samples
            while len(self.buffer) >= 160:
                chunk_size = (len(self.buffer) // 160) * 160
                chunk = self.buffer[:chunk_size]
                self.buffer = self.buffer[chunk_size:]

                try:
                    results = self.vad.detect_chunk(chunk)
                    # results is usually a list of StreamVadFrameResult
                    for res in results:
                        if res.is_speech:  # or check probability
                            print(f"🎤 Speech detected! Prob: {res.probability:.3f}")
                        # You can also accumulate speech segments here
                except Exception as e:
                    print(f"VAD error: {e}")

    # --- Create Manager and Attach VAD ---
    manager = AudioStreamManager(samplerate=16000, block_size=160)

    vad_observer = FireRedVADObserver()
    manager.add_observer(vad_observer)

    try:
        manager.start()
        print("Listening... Press Ctrl+C to stop")
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        manager.stop()
