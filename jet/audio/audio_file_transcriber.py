import os
import soundfile as sf
import numpy as np
import librosa
import logging

from typing import Optional, List
from faster_whisper import WhisperModel

from jet.logger import logger


class AudioFileTranscriber:
    def __init__(self, model_size: str = "small", sample_rate: Optional[int] = None):
        logger.setLevel(logging.DEBUG)
        self.model = WhisperModel(
            model_size, device="auto", compute_type="int8")
        self.sample_rate = sample_rate

    async def transcribe_from_file(self, file_path: str, output_dir: Optional[str] = None) -> Optional[str]:
        """Transcribe audio from a file.
        Args:
            file_path: Path to the audio file (e.g., WAV, MP3).
        Returns:
            Transcribed text or None if no speech is detected or an error occurs.
        """
        try:
            audio_data, file_sample_rate = sf.read(file_path)
            if self.sample_rate is not None and file_sample_rate != self.sample_rate:
                logger.info(
                    f"Resampling audio from {file_sample_rate} Hz to {self.sample_rate} Hz")
                audio_data = librosa.resample(
                    audio_data, orig_sr=file_sample_rate, target_sr=self.sample_rate)
            else:
                logger.info(
                    f"Using native sample rate: {file_sample_rate} Hz")
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            if audio_data.dtype != np.float32:
                if np.issubdtype(audio_data.dtype, np.integer):
                    audio_data = audio_data.astype(
                        np.float32) / np.iinfo(audio_data.dtype).max
                else:
                    audio_data = audio_data.astype(np.float32)
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
            if transcription and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_path = os.path.join(output_dir, base_name + ".txt")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(transcription)
                logger.info(f"Transcription saved to {output_path}")
            return transcription if transcription else None
        except FileNotFoundError:
            logger.error(f"Audio file not found: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error transcribing file {file_path}: {str(e)}")
            return None
        