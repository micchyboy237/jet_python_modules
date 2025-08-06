import pyaudio
import numpy as np
from faster_whisper import WhisperModel
from typing import Optional, Literal
import time
import logging
from queue import Queue
import threading
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class Transcriber:
    """A class for real-time audio transcription using Faster-Whisper."""

    def __init__(self, model_type: Literal["tiny", "base", "small", "medium", "large"] = "base", device: str = "auto"):
        """Initialize the transcriber with a Faster-Whisper model and audio settings."""
        self.model = WhisperModel(
            model_type, device=device, compute_type="float16")
        self.chunk_duration = 5  # seconds per audio chunk
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.audio_queue = Queue()
        self.is_running = False
        self.audio_thread = None
        self.pyaudio_instance = pyaudio.PyAudio()
        logging.info(
            f"Initialized Transcriber with Faster-Whisper model: {model_type}")

    def start_audio_stream(self) -> None:
        """Start capturing audio from the microphone and queue it for transcription."""
        stream = self.pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024
        )
        logging.info("Started audio stream.")

        try:
            while self.is_running:
                data = stream.read(
                    int(self.sample_rate * self.chunk_duration), exception_on_overflow=False)
                audio_np = np.frombuffer(
                    data, dtype=np.int16).astype(np.float32) / 32768.0
                self.audio_queue.put(audio_np)
        except Exception as e:
            logging.error(f"Error in audio stream: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            logging.info("Audio stream stopped.")

    def transcribe_audio(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe audio data using Faster-Whisper."""
        try:
            # Save audio to a temporary WAV file for Whisper processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file_name = temp_file.name
                # Write WAV file (simple WAV header for Faster-Whisper)
                import wave
                with wave.open(temp_file_name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes((audio * 32768).astype(np.int16).tobytes())

            # Transcribe using Faster-Whisper
            segments, _ = self.model.transcribe(temp_file_name, language="en")
            os.unlink(temp_file_name)  # Clean up temporary file
            transcription = " ".join(
                segment.text for segment in segments).strip()
            return transcription if transcription else None
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return None

    def start_transcription(self) -> None:
        """Start real-time transcription from the microphone."""
        self.is_running = True
        self.audio_thread = threading.Thread(target=self.start_audio_stream)
        self.audio_thread.start()
        logging.info(
            "Starting live transcription. Speak into the microphone...")

        try:
            while self.is_running:
                if not self.audio_queue.empty():
                    audio = self.audio_queue.get()
                    transcription = self.transcribe_audio(audio)
                    if transcription:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[{timestamp}] {transcription}")
                time.sleep(0.1)  # Prevent busy looping
        except KeyboardInterrupt:
            self.stop_transcription()
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            self.stop_transcription()

    def stop_transcription(self) -> None:
        """Stop the transcription process and clean up."""
        self.is_running = False
        if self.audio_thread:
            self.audio_thread.join()
        self.pyaudio_instance.terminate()
        logging.info("Transcription stopped.")


def main():
    """Main function to run the transcription program."""
    transcriber = Transcriber(model_type="base")
    try:
        transcriber.start_transcription()
    except KeyboardInterrupt:
        transcriber.stop_transcription()
        logging.info("Program terminated by user.")


if __name__ == "__main__":
    main()
