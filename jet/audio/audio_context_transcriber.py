import os
import soundfile as sf
import numpy as np
import librosa
import logging

from typing import Optional, List, Tuple

from jet.logger import logger
from jet.models.model_registry.transformers.speech_to_text.whisper_model_registry import WhisperModelRegistry


def ensure_stereo(audio: np.ndarray, expected_channels: int = 2) -> np.ndarray:
    """
    Ensure the audio array has the expected number of channels (e.g., stereo).
    If mono, duplicate the channel to match expected_channels.
    Args:
        audio: Input audio array (1D for mono, 2D for stereo or multi-channel).
        expected_channels: Number of channels to enforce (default: 2 for stereo).
    Returns:
        Audio array with the expected number of channels.
    """
    if audio.ndim == 1:
        # Mono audio: duplicate to create stereo
        audio = np.tile(audio[:, np.newaxis], (1, expected_channels))
    elif audio.ndim == 2 and audio.shape[1] != expected_channels:
        # If channel count doesn't match, take first channel and duplicate
        audio = np.tile(audio[:, :1], (1, expected_channels))
    return audio


class AudioContextTranscriber:
    def __init__(self, model_size: str = "small", sample_rate: Optional[int] = None):
        logger.setLevel(logging.DEBUG)
        registry = WhisperModelRegistry()
        self.model = registry.load_model(
            model_size, device="auto", compute_type="int8")
        self.sample_rate = sample_rate

    async def transcribe_with_context(
        self,
        file_path: str,
        prev_file_path: Optional[str] = None,
        next_file_path: Optional[str] = None,
        start_overlap_duration: float = 0.0,
        end_overlap_duration: float = 0.0,
        output_dir: Optional[str] = None,
        language="en",
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Transcribe an audio chunk with optional context from previous and next chunks.
        Returns transcription for the non-overlapping portion, start overlap, and end overlap.
        Args:
            file_path: Path to the current audio chunk file.
            prev_file_path: Path to the previous chunk file for start overlap context.
            next_file_path: Path to the next chunk file for end overlap context.
            start_overlap_duration: Duration of start overlap in seconds.
            end_overlap_duration: Duration of end overlap in seconds.
            output_dir: Directory to save transcription file.
            language: Language code for transcription. Default is "en".            
        Returns:
            Tuple of (non-overlap transcription, start overlap transcription, end overlap transcription).
        """
        try:
            audio_data, file_sample_rate = sf.read(file_path)
            logger.debug(
                f"Loaded chunk {file_path}, sample rate: {file_sample_rate}, samples: {len(audio_data)}")
            # Ensure audio_data is stereo
            audio_data = ensure_stereo(audio_data)
            if self.sample_rate is not None and file_sample_rate != self.sample_rate:
                logger.info(
                    f"Resampling chunk from {file_sample_rate} Hz to {self.sample_rate} Hz")
                audio_data = librosa.resample(
                    audio_data, orig_sr=file_sample_rate, target_sr=self.sample_rate)
                file_sample_rate = self.sample_rate

            # Initialize as empty 2D stereo
            prev_audio = np.zeros((0, 2), dtype=np.float32)
            if prev_file_path and start_overlap_duration > 0:
                prev_data, prev_sr = sf.read(prev_file_path)
                if prev_sr != file_sample_rate:
                    prev_data = librosa.resample(
                        prev_data, orig_sr=prev_sr, target_sr=file_sample_rate)
                # Ensure stereo for prev_audio
                prev_data = ensure_stereo(prev_data)
                start_overlap_samples = int(
                    start_overlap_duration * file_sample_rate)
                prev_audio = prev_data[-start_overlap_samples:] if len(
                    prev_data) >= start_overlap_samples else prev_data
                # Ensure stereo after slicing
                prev_audio = ensure_stereo(prev_audio)
                logger.debug(
                    f"Loaded {len(prev_audio)} samples from previous chunk {prev_file_path}")

            # Initialize as empty 2D stereo
            next_audio = np.zeros((0, 2), dtype=np.float32)
            if next_file_path and end_overlap_duration > 0:
                next_data, next_sr = sf.read(next_file_path)
                if next_sr != file_sample_rate:
                    next_data = librosa.resample(
                        next_data, orig_sr=next_sr, target_sr=file_sample_rate)
                # Ensure stereo for next_audio
                next_data = ensure_stereo(next_data)
                end_overlap_samples = int(
                    end_overlap_duration * file_sample_rate)
                next_audio = next_data[:end_overlap_samples] if len(
                    next_data) >= end_overlap_samples else next_data
                # Ensure stereo after slicing
                next_audio = ensure_stereo(next_audio)
                logger.debug(
                    f"Loaded {len(next_audio)} samples from next chunk {next_file_path}")
            combined_audio = np.concatenate([prev_audio, audio_data, next_audio]) if (
                len(prev_audio) > 0 or len(next_audio) > 0) else audio_data
            if combined_audio.dtype != np.float32:
                if np.issubdtype(combined_audio.dtype, np.integer):
                    combined_audio = combined_audio.astype(
                        np.float32) / np.iinfo(combined_audio.dtype).max
                else:
                    combined_audio = combined_audio.astype(np.float32)
            logger.debug(
                f"Combined audio length: {len(combined_audio)} samples, dtype: {combined_audio.dtype}")
            segments, info = self.model.transcribe(
                combined_audio,
                language=language,
                beam_size=1,  # Optimize for speed
                temperature=0,  # Deterministic output
                # beam_size=5,
                # vad_filter=True,
                # vad_parameters=dict(min_silence_duration_ms=500),
                # log_progress=True
            )
            start_overlap_samples = len(prev_audio)
            non_overlap_samples = len(audio_data)
            total_samples = len(combined_audio)
            end_overlap_samples = total_samples - start_overlap_samples - non_overlap_samples
            start_overlap_end_s = start_overlap_samples / \
                file_sample_rate if start_overlap_samples > 0 else 0.0
            non_overlap_end_s = (start_overlap_samples +
                                 non_overlap_samples) / file_sample_rate
            start_overlap_text = []
            non_overlap_text = []
            end_overlap_text = []
            for segment in segments:
                segment_start = segment.start
                segment_end = segment.end
                segment_text = segment.text.strip()
                if segment_end <= start_overlap_end_s:
                    start_overlap_text.append(segment_text)
                elif segment_start >= non_overlap_end_s:
                    end_overlap_text.append(segment_text)
                else:
                    segment_duration = segment_end - segment_start
                    if segment_duration == 0:
                        continue
                    start_overlap_fraction = max(
                        0.0, min(segment_end, start_overlap_end_s) - segment_start) / segment_duration
                    non_overlap_fraction = max(0.0, min(segment_end, non_overlap_end_s) - max(
                        segment_start, start_overlap_end_s)) / segment_duration
                    end_overlap_fraction = max(
                        0.0, segment_end - max(segment_start, non_overlap_end_s)) / segment_duration
                    max_fraction = max(
                        start_overlap_fraction, non_overlap_fraction, end_overlap_fraction)
                    if max_fraction == start_overlap_fraction and start_overlap_fraction > 0:
                        start_overlap_text.append(segment_text)
                    elif max_fraction == non_overlap_fraction and non_overlap_fraction > 0:
                        non_overlap_text.append(segment_text)
                    elif max_fraction == end_overlap_fraction and end_overlap_fraction > 0:
                        end_overlap_text.append(segment_text)
            start_overlap_transcription = " ".join(
                start_overlap_text).strip() if start_overlap_text else ""
            non_overlap_transcription = " ".join(
                non_overlap_text).strip() if non_overlap_text else ""
            end_overlap_transcription = " ".join(
                end_overlap_text).strip() if end_overlap_text else ""
            if non_overlap_transcription and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                try:
                    chunk_index = int(base_name.split(
                        '_')[-1]) if 'stream_chunk' in base_name else -1
                    output_filename = f"transcription_{chunk_index:05d}.txt" if chunk_index >= 0 else f"transcription_{base_name}.txt"
                except ValueError:
                    logger.debug(
                        f"Could not parse chunk index from {base_name}, using base name")
                    output_filename = f"transcription_{base_name}.txt"
                output_path = os.path.join(output_dir, output_filename)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(non_overlap_transcription)
                logger.info(
                    f"Non-overlap transcription saved to {output_path}")
            logger.debug(f"Transcriptions - Start overlap: '{start_overlap_transcription}', "
                         f"Non-overlap: '{non_overlap_transcription}', End overlap: '{end_overlap_transcription}'")
            return non_overlap_transcription, start_overlap_transcription, end_overlap_transcription
        except FileNotFoundError:
            logger.error(f"Audio file not found: {file_path}")
            return None, None, None
        except Exception as e:
            logger.error(f"Error transcribing chunk {file_path}: {str(e)}")
            return None, None, None
