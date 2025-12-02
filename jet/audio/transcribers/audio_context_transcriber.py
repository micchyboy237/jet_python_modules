"""
AudioContextTranscriber
Offline, context-aware transcriber for chunked audio files (e.g., streaming pipelines).
Loads a main audio chunk + optional overlapping regions from previous/next chunks,
concatenates them for contextual inference, then intelligently splits the transcription
back into three parts using precise timing and fractional segment allocation.
"""

import os
import soundfile as sf
import numpy as np
import librosa
import logging
from typing import Optional, Tuple
from jet.logger import logger
from jet.models.model_registry.transformers.speech_to_text.whisper_model_registry import WhisperModelRegistry


def ensure_stereo(audio: np.ndarray, expected_channels: int = 2) -> np.ndarray:
    """Ensure audio has expected_channels (e.g., stereo) by duplicating mono if needed."""
    if audio.ndim == 1:
        audio = np.tile(audio[:, np.newaxis], (1, expected_channels))
    elif audio.ndim == 2 and audio.shape[1] != expected_channels:
        audio = np.tile(audio[:, :1], (1, expected_channels))
    return audio


class AudioContextTranscriber:
    def __init__(self, model_size: str = "large-v3", sample_rate: Optional[int] = None, device: str = "cpu", compute_type: str = "int8"):
        logger.setLevel(logging.DEBUG)
        registry = WhisperModelRegistry()
        self.model = registry.load_model(model_size, device=device, compute_type=compute_type)
        self.sample_rate = sample_rate

    def transcribe_with_context(  # ← REMOVED `async` — now synchronous
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
        Synchronous version — runs Whisper inference on current thread.
        Must be called via run_in_executor() to avoid blocking the event loop.
        """
        try:
            audio_data, file_sample_rate = sf.read(file_path)
            logger.debug(
                f"Loaded chunk {file_path}, sample rate: {file_sample_rate}, samples: {len(audio_data)}")
            audio_data = ensure_stereo(audio_data)

            if self.sample_rate is not None and file_sample_rate != self.sample_rate:
                logger.info(
                    f"Resampling chunk from {file_sample_rate} Hz to {self.sample_rate} Hz")
                audio_data = librosa.resample(
                    audio_data, orig_sr=file_sample_rate, target_sr=self.sample_rate)
                file_sample_rate = self.sample_rate

            prev_audio = np.zeros((0, 2), dtype=np.float32)
            if prev_file_path and start_overlap_duration > 0:
                prev_data, prev_sr = sf.read(prev_file_path)
                if prev_sr != file_sample_rate:
                    prev_data = librosa.resample(
                        prev_data, orig_sr=prev_sr, target_sr=file_sample_rate)
                prev_data = ensure_stereo(prev_data)
                start_overlap_samples = int(start_overlap_duration * file_sample_rate)
                prev_audio = prev_data[-start_overlap_samples:] if len(prev_data) >= start_overlap_samples else prev_data
                logger.debug(f"Loaded {len(prev_audio)} samples from previous chunk")

            next_audio = np.zeros((0, 2), dtype=np.float32)
            if next_file_path and end_overlap_duration > 0:
                next_data, next_sr = sf.read(next_file_path)
                if next_sr != file_sample_rate:
                    next_data = librosa.resample(
                        next_data, orig_sr=next_sr, target_sr=file_sample_rate)
                next_data = ensure_stereo(next_data)
                end_overlap_samples = int(end_overlap_duration * file_sample_rate)
                next_audio = next_data[:end_overlap_samples] if len(next_data) >= end_overlap_samples else next_data
                logger.debug(f"Loaded {len(next_audio)} samples from next chunk")

            combined_audio = np.concatenate([prev_audio, audio_data, next_audio]) if (
                len(prev_audio) > 0 or len(next_audio) > 0) else audio_data

            if combined_audio.dtype != np.float32:
                if np.issubdtype(combined_audio.dtype, np.integer):
                    combined_audio = combined_audio.astype(np.float32) / np.iinfo(combined_audio.dtype).max
                else:
                    combined_audio = combined_audio.astype(np.float32)

            logger.debug(f"Combined audio length: {len(combined_audio)} samples")

            if combined_audio.ndim > 1:
                audio_mono = combined_audio.mean(axis=1) if combined_audio.shape[1] == 2 else combined_audio[:, 0]
            else:
                audio_mono = combined_audio

            if audio_mono.max() > 1.0 or audio_mono.min() < -1.0:
                audio_mono = audio_mono / np.max(np.abs(audio_mono))

            logger.debug(f"Transcribing mono audio of {len(audio_mono)} samples")
            segments, info = self.model.transcribe(
                audio_mono,
                language=language,
                task="translate",

                # Decoding: Maximum accuracy
                beam_size=10,
                patience=2.0,
                temperature=0.0,
                length_penalty=1.0,
                best_of=1,
                log_prob_threshold=-0.5,

                # Context & consistency
                condition_on_previous_text=True,

                # Japanese punctuation handling
                prepend_punctuations="\"'“¿([{-『「（［",
                append_punctuations="\"'.。,，!！?？:：”)]}、。」」！？",

                # Clean input
                vad_filter=True,
                vad_parameters=None,

                # Output options
                without_timestamps=False,
                word_timestamps=True,
                chunk_length=30,
                log_progress=True,
            )

            start_overlap_samples = len(prev_audio)
            non_overlap_samples = len(audio_data)
            total_samples = len(combined_audio)
            end_overlap_samples = total_samples - start_overlap_samples - non_overlap_samples

            start_overlap_end_s = start_overlap_samples / file_sample_rate if start_overlap_samples > 0 else 0.0
            non_overlap_end_s = (start_overlap_samples + non_overlap_samples) / file_sample_rate

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
                    segment_duration = segment_end - segment_start or 1e-6
                    start_overlap_fraction = max(0.0, min(segment_end, start_overlap_end_s) - segment_start) / segment_duration
                    non_overlap_fraction = max(0.0, min(segment_end, non_overlap_end_s) - max(segment_start, start_overlap_end_s)) / segment_duration
                    end_overlap_fraction = max(0.0, segment_end - max(segment_start, non_overlap_end_s)) / segment_duration

                    max_fraction = max(start_overlap_fraction, non_overlap_fraction, end_overlap_fraction)
                    if max_fraction == start_overlap_fraction and start_overlap_fraction > 0:
                        start_overlap_text.append(segment_text)
                    elif max_fraction == non_overlap_fraction and non_overlap_fraction > 0:
                        non_overlap_text.append(segment_text)
                    elif max_fraction == end_overlap_fraction and end_overlap_fraction > 0:
                        end_overlap_text.append(segment_text)

            start_overlap_transcription = " ".join(start_overlap_text).strip() if start_overlap_text else ""
            non_overlap_transcription = " ".join(non_overlap_text).strip() if non_overlap_text else ""
            end_overlap_transcription = " ".join(end_overlap_text).strip() if end_overlap_text else ""

            if non_overlap_transcription and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                try:
                    chunk_index = int(base_name.split('_')[-1]) if 'stream_chunk' in base_name else -1
                    output_filename = f"transcription_{chunk_index:05d}.txt" if chunk_index >= 0 else f"transcription_{base_name}.txt"
                except ValueError:
                    output_filename = f"transcription_{base_name}.txt"
                output_path = os.path.join(output_dir, output_filename)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(non_overlap_transcription)
                logger.info(f"Non-overlap transcription saved to {output_path}")

            logger.debug(f"Transcriptions - Start: '{start_overlap_transcription}', "
                         f"Non-overlap: '{non_overlap_transcription}', End: '{end_overlap_transcription}'")

            return non_overlap_transcription, start_overlap_transcription, end_overlap_transcription

        except FileNotFoundError:
            logger.error(f"Audio file not found: {file_path}")
            return None, None, None
        except Exception as e:
            logger.error(f"Error transcribing chunk {file_path}: {str(e)}")
            return None, None, None