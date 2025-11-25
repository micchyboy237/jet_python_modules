"""
Real-time speech-to-text transcriber using system microphone audio.

Features:
- Captures live audio from the default microphone using sounddevice
- Uses faster-whisper (CTranslate2) with int8 quantization for fast, accurate transcription
- Built-in VAD (voice activity detection) to stop recording after ~500ms of silence
- Runs continuously, printing new transcriptions as speech is detected
- Works on Windows, macOS, Linux, and even Pyodide/Emscripten (browser via WebAssembly)

Setup & Installation (one-time):
    pip install sounddevice numpy faster-whisper

    # On Windows (usually not needed):
    # pip install sounddevice --only-binary=:all:

    # On macOS you may need portaudio:
    brew install portaudio

    # On Linux (Debian/Ubuntu):
    sudo apt-get install portaudio19-dev
    pip install sounddevice

Run:
    python audio_transcriber.py

    Speak into your microphone. When you pause for ~0.5 seconds, the detected speech
    will be transcribed and printed. Press Ctrl+C to stop.

Customization (optional):
    AudioSystemTranscriber(
        model_size="large-v3",   # Options: tiny, base, small, medium, large-v2, large-v3
        sample_rate=16000,       # Must be 16000 for Whisper models
        chunk_duration=1.0       # Seconds per audio block (1.0 works well)
    )

Note:
    - First run downloads the selected Whisper model (~1-7 GB depending on size)
    - Uses "auto" device → CUDA if available, otherwise CPU with int8 for speed
"""

import sounddevice as sd
import numpy as np
import asyncio
import platform

from typing import Any, Optional, List
from faster_whisper import WhisperModel

from jet.logger import logger

class AudioSystemTranscriber:
    def __init__(self, model_size: str = "turbo", sample_rate: int = 16000, chunk_duration: float = 1.0):
        """
        Initializes faster-whisper model (int8, auto device)
        Sets sample rate (16 kHz default), chunk duration and calculates chunk_samples
        Detects and stores the default input device ID
        Initializes empty frame buffer and silence counter
        """
        logger.basicConfig()
        logger.getLogger("faster_whisper").setLevel(logger.DEBUG)
        self.model = WhisperModel(
            model_size, device="auto", compute_type="int8")
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.frames: List[np.ndarray] = []
        self.silent_count = 0
        self.input_device = self._get_default_input_device()

    def _get_default_input_device(self) -> int:
        """
        Queries sounddevice for the current default input device
        Validates that it has input channels
        Logs the selected device name and returns its ID
        Raises ValueError with detailed message on failure
        """
        try:
            # Get default input device ID
            default_device = sd.default.device[0]
            device_info = sd.query_devices(default_device)
            if device_info['max_input_channels'] > 0:
                logger.info(
                    f"Selected default input device: {device_info['name']} (ID: {default_device})")
                return default_device
            raise ValueError("Default input device has no input channels.")
        except Exception as e:
            raise ValueError(
                f"Failed to find a valid default input device: {str(e)}")

    def callback(self, indata: np.ndarray, frames: int, time: Any, status: sd.CallbackFlags) -> None:
        """
        def callback(self, indata: np.ndarray, frames: int, time: Any, status: sd.CallbackFlags) -> None:
            # sounddevice InputStream callback
            # Logs any stream errors/warnings
            # Appends a copy of the incoming audio chunk (int16) to self.frames
        """
        if status:
            logger.warning(f"Stream status: {status}")
            return
        self.frames.append(indata.copy())

    async def capture_and_transcribe(self) -> Optional[str]:
        """
        async def capture_and_transcribe(self) -> Optional[str]:
        Clears previous buffers
        Opens a blocking InputStream with the configured parameters
        Keeps the stream alive with short async sleeps until the callback or VAD stops it
        Concatenates collected frames → float32 normalized numpy array
        Runs faster-whisper transcription with:
            language="en", beam_size=5, VAD filter enabled, 500 ms min silence
        Joins segment texts into a single transcription string
        Returns the transcription or None if empty / no speech
        """
        self.frames = []
        self.silent_count = 0

        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16',
                                blocksize=self.chunk_samples, callback=self.callback,
                                device=self.input_device):
                logger.info("Listening for speech...")
                while True:
                    await asyncio.sleep(0.1)
        except sd.CallbackStop:
            logger.debug(
                "Audio capture stopped due to sufficient frames or silence.")
            pass

        if not self.frames:
            return None

        audio_data = np.concatenate(self.frames).astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            audio_data,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            log_progress=True
        )
        transcription = " ".join(segment.text for segment in segments).strip()
        return transcription if transcription else None


async def main():
    """
    Instantiates AudioSystemTranscriber with default settings
    Infinite loop that awaits a new transcription on every speech segment
    Prints the result or "No speech detected."
    Catches KeyboardInterrupt for clean shutdown
    """
    transcriber = AudioSystemTranscriber()
    while True:
        try:
            transcription = await transcriber.capture_and_transcribe()
            if transcription:
                print(f"Transcription: {transcription}")
            else:
                print("No speech detected.")
            await asyncio.sleep(1.0)
        except KeyboardInterrupt:
            print("\nStopping transcription.")
            break

"""
if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
# Platform-specific asyncio entry point (standard Python vs Pyodide/Emscripten)
"""
if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
