import soundfile as sf
import numpy as np
import librosa
import logging

from typing import Optional, List
from faster_whisper import WhisperModel

from jet.logger import logger


class AudioFileTranscriber:
    def __init__(self, model_size: str = "small", sample_rate: int = 16000):
        logging.basicConfig()
        logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
        self.model = WhisperModel(
            model_size, device="auto", compute_type="int8")
        self.sample_rate = sample_rate

    async def transcribe_from_file(self, file_path: str) -> Optional[str]:
        """Transcribe audio from a file.

        Args:
            file_path: Path to the audio file (e.g., WAV, MP3).

        Returns:
            Transcribed text or None if no speech is detected or an error occurs.
        """
        try:
            # Read audio file and resample if necessary
            audio_data, file_sample_rate = sf.read(file_path)
            if file_sample_rate != self.sample_rate:
                logger.info(
                    f"Resampling audio from {file_sample_rate} Hz to {self.sample_rate} Hz")
                audio_data = librosa.resample(
                    audio_data, orig_sr=file_sample_rate, target_sr=self.sample_rate)

            # Ensure audio is mono
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Normalize to float32 in range [-1.0, 1.0]
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(
                    np.float32) / np.iinfo(audio_data.dtype).max

            # Transcribe using the same settings as capture_and_transcribe
            segments, _ = self.model.transcribe(
                audio_data,
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                log_progress=True
            )
            transcription = " ".join(
                segment.text for segment in segments).strip()
            logger.info(f"Transcription completed for file: {file_path}")
            return transcription if transcription else None

        except FileNotFoundError:
            logger.error(f"Audio file not found: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error transcribing file {file_path}: {str(e)}")
            return None
