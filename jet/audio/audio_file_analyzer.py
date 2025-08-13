import os
import soundfile as sf
import librosa
import numpy as np
from typing import Dict, Optional, Tuple, Literal
from jet.logger import logger


class AudioFileAnalyzer:
    """A class to extract metadata and audio features from an audio file."""

    def __init__(self, file_path: str):
        """Initialize the analyzer with the path to the audio file.

        Args:
            file_path: Path to the audio file (e.g., WAV, MP3).
        """
        self.file_path = file_path
        logger.setLevel('DEBUG')

    def get_basic_metadata(self) -> Dict[str, any]:
        """Extract basic metadata from the audio file.

        Returns:
            Dictionary containing file format, sample rate, channels, duration, file size, and bit depth.
        """
        try:
            with sf.SoundFile(self.file_path) as audio:
                metadata = {
                    "file_path": self.file_path,
                    "file_format": audio.format,
                    "sample_rate": audio.samplerate,
                    "channels": audio.channels,
                    "duration_s": len(audio) / audio.samplerate,
                    "file_size_bytes": os.path.getsize(self.file_path),
                    "bit_depth": self._infer_bit_depth(audio.subtype)
                }
                logger.debug(f"Extracted basic metadata: {metadata}")
                return metadata
        except Exception as e:
            logger.error(
                f"Error extracting basic metadata from {self.file_path}: {str(e)}")
            return {}

    def _infer_bit_depth(self, subtype: str) -> Optional[int]:
        """Infer bit depth from the audio subtype.

        Args:
            subtype: Audio subtype from soundfile (e.g., PCM_16).

        Returns:
            Bit depth as an integer, or None if unknown.
        """
        subtype_map = {
            'PCM_16': 16,
            'PCM_24': 24,
            'PCM_32': 32,
            'FLOAT': 32,
            'DOUBLE': 64
        }
        bit_depth = subtype_map.get(subtype, None)
        logger.debug(f"Inferred bit depth for subtype {subtype}: {bit_depth}")
        return bit_depth

    def get_audio_features(self) -> Dict[str, any]:
        """Extract audio features such as pitch, tempo, and spectral characteristics.

        Returns:
            Dictionary containing pitch, tempo, spectral centroid, and RMS energy.
        """
        try:
            audio_data, sample_rate = sf.read(self.file_path)
            logger.debug(
                f"Loaded audio: samples={len(audio_data)}, sample_rate={sample_rate}")
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)  # Convert to mono
                logger.debug(
                    f"Converted to mono, new shape: {audio_data.shape}")
            if audio_data.dtype != np.float32:
                if np.issubdtype(audio_data.dtype, np.integer):
                    audio_data = audio_data.astype(
                        np.float32) / np.iinfo(audio_data.dtype).max
                    logger.debug(
                        f"Converted integer audio to float32, max value: {np.max(audio_data)}")
                else:
                    audio_data = audio_data.astype(np.float32)
                    logger.debug(
                        f"Converted to float32, dtype was: {audio_data.dtype}")

            # Extract pitch using YIN algorithm
            pitches, magnitudes = librosa.piptrack(
                y=audio_data, sr=sample_rate)
            mean_pitch = np.mean(pitches[pitches > 0]) if np.any(
                pitches > 0) else 0.0
            logger.debug(
                f"Pitch extraction: mean_pitch={mean_pitch}, non-zero pitches={np.sum(pitches > 0)}")

            # Extract tempo
            tempo, beats = librosa.beat.beat_track(
                y=audio_data, sr=sample_rate)
            logger.debug(
                f"Tempo extraction: tempo={tempo}, num_beats={len(beats)}")

            # Extract spectral centroid
            spectral_centroid = np.mean(
                librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))
            logger.debug(f"Spectral centroid: {spectral_centroid}")

            # Extract RMS energy
            rms = np.mean(librosa.feature.rms(y=audio_data))
            logger.debug(f"RMS energy: {rms}")

            features = {
                "mean_pitch_hz": float(mean_pitch),
                "tempo_bpm": float(tempo),
                "spectral_centroid_hz": float(spectral_centroid),
                "rms_energy": float(rms)
            }
            logger.debug(f"Extracted audio features: {features}")
            return features
        except Exception as e:
            logger.error(
                f"Error extracting audio features from {self.file_path}: {str(e)}")
            return {}

    def analyze(self) -> Dict[str, any]:
        """Combine basic metadata and audio features into a single analysis result.

        Returns:
            Dictionary containing all extracted metadata and features.
        """
        result = {
            "metadata": self.get_basic_metadata(),
            "features": self.get_audio_features()
        }
        logger.info(f"Analysis completed for {self.file_path}: {result}")
        return result
