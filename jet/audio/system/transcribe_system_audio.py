import sounddevice as sd
import numpy as np
from typing import Any, Optional, List
from faster_whisper import WhisperModel
import asyncio
import platform
import logging


class AudioTranscriber:
    def __init__(self, model_size: str = "small", sample_rate: int = 16000, chunk_duration: float = 1.0):
        logging.basicConfig()
        logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
        self.model = WhisperModel(
            model_size, device="auto", compute_type="int8")
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.frames: List[np.ndarray] = []
        self.silent_count = 0
        self.input_device = self._get_blackhole_device()

    def _get_blackhole_device(self) -> int:
        """Find the BlackHole input device ID."""
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if "BlackHole" in device['name'] and device['max_input_channels'] > 0:
                logging.info(
                    f"Selected input device: {device['name']} (ID: {i})")
                return i
        raise ValueError(
            "BlackHole device not found. Ensure BlackHole is installed and configured.")

    def callback(self, indata: np.ndarray, frames: int, time: Any, status: sd.CallbackFlags) -> None:
        if status:
            logging.warning(f"Stream status: {status}")
            return
        self.frames.append(indata.copy())

    async def capture_and_transcribe(self) -> Optional[str]:
        self.frames = []
        self.silent_count = 0

        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16',
                                blocksize=self.chunk_samples, callback=self.callback,
                                device=self.input_device):
                logging.info("Listening for speech...")
                while True:
                    await asyncio.sleep(0.1)
        except sd.CallbackStop:
            logging.debug(
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
    transcriber = AudioTranscriber()
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

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
