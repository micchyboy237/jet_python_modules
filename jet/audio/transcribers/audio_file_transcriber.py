import os
import soundfile as sf
import numpy as np
import librosa
import logging

from typing import Optional

from jet.logger import logger
from jet.models.model_registry.transformers.speech_to_text.whisper_model_registry import WhisperModelRegistry, WhisperModelsType


class AudioFileTranscriber:
    def __init__(self, model_size: WhisperModelsType = "small", sample_rate: Optional[int] = None):
        logger.setLevel(logging.DEBUG)
        registry = WhisperModelRegistry()
        self.model = registry.load_model(
            model_size, device="auto", compute_type="int8")
        self.sample_rate = sample_rate

    async def transcribe_from_file(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        language: str = "en"
    ) -> Optional[str]:
        """Transcribe a single audio file using translate task (clean English output)."""
        try:
            audio_data, file_sample_rate = sf.read(file_path)
            logger.debug(
                f"Loaded audio file {file_path}, "
                f"sample rate: {file_sample_rate}, "
                f"shape: {audio_data.shape}, dtype: {audio_data.dtype}"
            )

            # Resample if target sample rate is specified
            if self.sample_rate is not None and file_sample_rate != self.sample_rate:
                logger.info(f"Resampling from {file_sample_rate} Hz → {self.sample_rate} Hz")
                audio_data = librosa.resample(
                    audio_data, orig_sr=file_sample_rate, target_sr=self.sample_rate
                )
                file_sample_rate = self.sample_rate
            else:
                logger.info(f"Using native sample rate: {file_sample_rate} Hz")

            # Convert to mono (1D) — faster-whisper requires this
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)  # Downmix stereo/multi-channel
                logger.debug("Downmixed to mono")

            # Normalize to float32 in [-1.0, 1.0]
            if not np.issubdtype(audio_data.dtype, np.floating):
                audio_data = audio_data.astype(np.float32)
                max_val = np.iinfo(audio_data.dtype).max
                audio_data /= max_val
                logger.debug("Converted integer audio to float32 and normalized")
            elif audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Final sanity check
            if audio_data.ndim != 1:
                raise ValueError(f"Audio must be 1D after processing, got shape {audio_data.shape}")
            if len(audio_data) == 0:
                logger.warning("Empty audio array after processing")
                return None

            logger.debug(f"Transcribing {len(audio_data)} samples (mono, float32)")

            segments, info = self.model.transcribe(
                audio_data,                    # Must be 1D mono float32
                language=language,
                task="translate",              # Forces clean English output
                beam_size=7,                   # Better accuracy than 1
                best_of=5,
                temperature=(0.0, 0.2),        # Slight sampling for better fluency
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=700),
                word_timestamps=False,         # Not needed here
            )

            transcription = " ".join(segment.text.strip() for segment in segments).strip()

            if not transcription:
                logger.info("No speech detected in audio")
                return None

            # Save to file if output_dir provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                try:
                    index_part = base_name.split('_')[-1]
                    chunk_index = int(index_part) if index_part.isdigit() else -1
                except (IndexError, ValueError):
                    chunk_index = -1

                if chunk_index >= 0:
                    output_filename = f"transcription_{chunk_index:05d}.txt"
                else:
                    output_filename = f"transcription_{base_name}.txt"

                output_path = os.path.join(output_dir, output_filename)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(transcription)
                logger.info(f"Transcription saved → {output_path}")

            logger.debug(f"Final transcription: {transcription[:200]}{'...' if len(transcription) > 200 else ''}")
            return transcription

        except FileNotFoundError:
            logger.error(f"Audio file not found: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error transcribing file {file_path}: {type(e).__name__}: {e}")
            return None
