# jet_python_modules/jet/libs/sherpa_onnx/audio_tagger.py

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import sherpa_onnx
from jet.audio.audio_types import AudioInput
from jet.audio.utils.loader import load_audio

logger = logging.getLogger(__name__)

BASE_DIR = Path("~/.cache/pretrained_models/sherpa-onnx").expanduser().resolve()
AUDIO_TAGGING_MODEL = (
    BASE_DIR / "sherpa-onnx-zipformer-audio-tagging-2024-04-09/model.onnx"
)
CLASS_LABELS_INDICES_CSV = (
    BASE_DIR / "sherpa-onnx-zipformer-audio-tagging-2024-04-09/class_labels_indices.csv"
)


class AudioTagger:
    """
    Reusable audio tagging class using Sherpa-ONNX models.

    Features:
    - Tags audio with type AudioInput (reuses load_audio)
    - Checks if audio contains high-probability speech

    Example:
        tagger = AudioTagger()
        results = tagger.tag_audio("path/to/audio.wav")
        is_speech = tagger.contains_speech("path/to/audio.wav")
    """

    # Default model paths
    DEFAULT_BASE_DIR = (
        Path("~/.cache/pretrained_models/sherpa-onnx").expanduser().resolve()
    )
    DEFAULT_MODEL_PATH = (
        DEFAULT_BASE_DIR / "sherpa-onnx-zipformer-audio-tagging-2024-04-09/model.onnx"
    )
    DEFAULT_LABELS_PATH = (
        DEFAULT_BASE_DIR
        / "sherpa-onnx-zipformer-audio-tagging-2024-04-09/class_labels_indices.csv"
    )

    # Speech-related class names to check
    SPEECH_CLASS_NAMES = [
        "Speech",
        "Male speech, man speaking",
        "Female speech, woman speaking",
        "Child speech, kid speaking",
        "Conversation",
        "Narration, monologue",
    ]

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = AUDIO_TAGGING_MODEL,
        labels_path: Optional[Union[str, Path]] = CLASS_LABELS_INDICES_CSV,
        top_k: int = 5,
        num_threads: int = 1,
        provider: str = "cpu",
        debug: bool = False,
        speech_prob_threshold: float = 0.5,
        speech_top_n: int = 3,
    ):
        """
        Initialize the AudioTagger with model configuration.

        Args:
            model_path: Path to ONNX model file
            labels_path: Path to class labels CSV
            top_k: Number of top predictions to return
            num_threads: Number of CPU threads
            provider: Computation provider ("cpu", "cuda", etc.)
            debug: Enable debug logging for Sherpa-ONNX
            speech_prob_threshold: Minimum probability to consider as speech
            speech_top_n: Check top N predictions for speech classes
        """
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        self.labels_path = (
            Path(labels_path) if labels_path else self.DEFAULT_LABELS_PATH
        )
        self.top_k = top_k
        self.num_threads = num_threads
        self.provider = provider
        self.debug = debug
        self.speech_prob_threshold = speech_prob_threshold
        self.speech_top_n = speech_top_n

        # Lazy-loaded tagger instance
        self._tagger: Optional[sherpa_onnx.AudioTagging] = None
        self._labels_map: Optional[Dict[int, str]] = None

        logger.info(f"AudioTagger initialized with model: {self.model_path}")
        logger.info(f"Labels file: {self.labels_path}")
        logger.info(f"Speech threshold: {self.speech_prob_threshold}")

    def _validate_model_files(self) -> None:
        """Validate that required model files exist."""
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                "Download from: https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models"
            )
        if not self.labels_path.is_file():
            raise FileNotFoundError(
                f"Labels file not found: {self.labels_path}\n"
                "Download from: https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models"
            )
        logger.debug(f"Model files validated: {self.model_path}, {self.labels_path}")

    def _load_labels(self) -> Dict[int, str]:
        """Load class labels from CSV file."""
        import csv

        labels = {}
        with open(self.labels_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    try:
                        index = int(row[0])
                        labels[index] = row[1].strip('"').strip()
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Skipping invalid label row: {row}, error: {e}")

        logger.debug(f"Loaded {len(labels)} class labels")
        return labels

    @property
    def tagger(self) -> sherpa_onnx.AudioTagging:
        """Lazy-load the Sherpa-ONNX AudioTagging instance."""
        if self._tagger is None:
            self._validate_model_files()

            config = sherpa_onnx.AudioTaggingConfig(
                model=sherpa_onnx.AudioTaggingModelConfig(
                    zipformer=sherpa_onnx.OfflineZipformerAudioTaggingModelConfig(
                        model=str(self.model_path),
                    ),
                    num_threads=self.num_threads,
                    debug=self.debug,
                    provider=self.provider,
                ),
                labels=str(self.labels_path),
                top_k=self.top_k,
            )

            if not config.validate():
                raise ValueError(f"Invalid AudioTaggingConfig: {config}")

            logger.info(f"Creating AudioTagger with config:\n{config}")
            self._tagger = sherpa_onnx.AudioTagging(config)
            self._labels_map = self._load_labels()

        return self._tagger

    @property
    def labels_map(self) -> Dict[int, str]:
        """Get the labels mapping."""
        if self._labels_map is None:
            self._labels_map = self._load_labels()
        return self._labels_map

    def tag_audio(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Tag audio with predicted labels and probabilities.

        Args:
            audio: Audio input (file path, bytes, numpy array, or torch tensor)
            sample_rate: Sample rate for raw audio data (ignored for file paths)

        Returns:
            List of dicts with keys: index, name, class_index, prob

        Example:
            >>> tagger = AudioTagger()
            >>> results = tagger.tag_audio("speech.wav")
            >>> for r in results:
            ...     print(f"{r['name']}: {r['prob']:.3f}")
        """
        start_time = time.time()
        logger.info(f"Tagging audio input of type: {type(audio).__name__}")

        # Load and normalize audio using the existing utility
        try:
            waveform, actual_sr = load_audio(audio, sr=sample_rate or 16000, mono=True)
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise

        logger.debug(
            f"Audio loaded: shape={waveform.shape}, sr={actual_sr}, "
            f"dtype={waveform.dtype}, "
            f"min={waveform.min():.4f}, max={waveform.max():.4f}"
        )

        # Create stream and process
        try:
            stream = self.tagger.create_stream()
            stream.accept_waveform(sample_rate=actual_sr, waveform=waveform)
            raw_results = self.tagger.compute(stream)
        except Exception as e:
            logger.error(f"Audio tagging failed: {e}")
            raise

        # Convert results to dictionaries
        results = []
        for i, event in enumerate(raw_results):
            result = {
                "index": i,
                "name": getattr(event, "name", "Unknown"),
                "class_index": getattr(event, "index", -1),
                "prob": getattr(event, "prob", 0.0),
            }
            results.append(result)
            logger.debug(
                f"Result {i}: {result['name']} (idx={result['class_index']}) "
                f"- prob={result['prob']:.4f}"
            )

        # Log performance metrics
        elapsed = time.time() - start_time
        audio_duration = len(waveform) / actual_sr if actual_sr > 0 else 0
        rtf = elapsed / audio_duration if audio_duration > 0 else float("inf")

        logger.info(
            f"Tagging complete: {len(results)} results, "
            f"duration={audio_duration:.2f}s, "
            f"elapsed={elapsed:.3f}s, RTF={rtf:.3f}"
        )

        return results

    def contains_speech(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
        prob_threshold: Optional[float] = None,
        top_n: Optional[int] = None,
    ) -> bool:
        """
        Check if audio contains speech with high probability.

        Args:
            audio: Audio input (file path, bytes, numpy array, or torch tensor)
            sample_rate: Sample rate for raw audio data
            prob_threshold: Override default speech probability threshold
            top_n: Override default number of top predictions to check

        Returns:
            True if speech is detected with probability >= threshold

        Example:
            >>> tagger = AudioTagger()
            >>> if tagger.contains_speech("meeting.wav"):
            ...     print("Speech detected!")
        """
        threshold = (
            prob_threshold if prob_threshold is not None else self.speech_prob_threshold
        )
        n_to_check = top_n if top_n is not None else self.speech_top_n

        logger.info(f"Checking for speech (threshold={threshold}, top_n={n_to_check})")

        try:
            # Reuse tag_audio to get predictions
            results = self.tag_audio(audio, sample_rate=sample_rate)
        except Exception as e:
            logger.error(f"Speech detection failed during tagging: {e}")
            return False

        if not results:
            logger.warning("No tagging results returned")
            return False

        # Check top N results for speech-related classes
        for result in results[:n_to_check]:
            name = result.get("name", "")
            prob = result.get("prob", 0.0)

            logger.debug(f"Checking: '{name}' (prob={prob:.4f}) against speech classes")

            # Check if this class is speech-related and meets threshold
            if name in self.SPEECH_CLASS_NAMES and prob >= threshold:
                logger.info(f"Speech detected: '{name}' with probability {prob:.4f}")
                return True

        # Additional check: if "Speech" is the primary class with highest probability
        top_result = results[0]
        top_name = top_result.get("name", "")
        top_prob = top_result.get("prob", 0.0)

        if top_name == "Speech" and top_prob >= threshold:
            logger.info(
                f"Speech detected as top result: '{top_name}' with probability {top_prob:.4f}"
            )
            return True

        logger.info(
            f"No speech detected. Top result: '{top_name}' (prob={top_prob:.4f})"
        )
        return False

    def get_speech_probability(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
    ) -> float:
        """
        Get the maximum speech probability from tagging results.

        Args:
            audio: Audio input
            sample_rate: Sample rate for raw audio data

        Returns:
            Maximum probability for any speech-related class (0.0 to 1.0)
        """
        try:
            results = self.tag_audio(audio, sample_rate=sample_rate)
        except Exception as e:
            logger.error(f"Failed to get speech probability: {e}")
            return 0.0

        # Find maximum probability among speech-related classes
        max_speech_prob = 0.0
        for result in results:
            if result.get("name", "") in self.SPEECH_CLASS_NAMES:
                prob = result.get("prob", 0.0)
                if prob > max_speech_prob:
                    max_speech_prob = prob
                    logger.debug(
                        f"New max speech prob: {prob:.4f} for '{result['name']}'"
                    )

        return max_speech_prob

    def reset(self) -> None:
        """Reset the tagger instance (useful for testing or model updates)."""
        self._tagger = None
        self._labels_map = None
        logger.info("AudioTagger reset")

    def save_results(
        self,
        results: List[Dict[str, Any]],
        output_path: Union[str, Path],
    ) -> None:
        """
        Save tagging results to a JSON file.

        Args:
            results: List of result dictionaries from tag_audio
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    audio_patb = "/Users/jethroestrada/.cache/files/audio/sub_audio/start_5s_recording_3_speakers.wav"

    # Basic usage
    tagger = AudioTagger()

    # Tag an audio file
    results = tagger.tag_audio(audio_patb)

    # Check for speech
    if tagger.contains_speech(audio_patb, prob_threshold=0.6):
        print("This recording contains speech!")

    # Get exact speech probability
    speech_prob = tagger.get_speech_probability(audio_patb)
    print(f"Speech probability: {speech_prob:.2%}")

    # # Custom model paths
    # custom_tagger = AudioTagger(
    #     model_path="/path/to/model.onnx",
    #     labels_path="/path/to/labels.csv",
    #     top_k=10,
    #     speech_prob_threshold=0.6,
    # )
