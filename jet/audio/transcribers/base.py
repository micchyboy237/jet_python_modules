from __future__ import annotations

import os
import ctranslate2
import numpy as np
import librosa
import transformers
from typing import Literal, Tuple

QuantizedModelSizes = Literal[
    "tiny", "base", "small", "medium", "large-v2", "large-v3"
]


def load_whisper_ct2_model(
    model_size: QuantizedModelSizes,
    model_dir: str,
) -> Tuple[ctranslate2.models.Whisper, transformers.WhisperProcessor]:
    """Load quantized CTranslate2 Whisper model + processor."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model not found: {model_dir}\n"
            f"Convert with:\n"
            f"  ct2-transformers-converter --model openai/whisper-{model_size} "
            f"--output_dir {model_dir} --quantization int8_float16"
        )
    processor = transformers.WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}")
    model = ctranslate2.models.Whisper(model_dir)
    return model, processor


def load_audio(
    audio_path: str,
    sr: int = 16_000,
    mono: bool = True,
) -> np.ndarray:
    """Load audio file → float32 numpy array at 16kHz."""
    audio, _ = librosa.load(audio_path, sr=sr, mono=mono)
    return audio.astype(np.float32)


def preprocess_audio(
    audio: np.ndarray,
    processor: transformers.WhisperProcessor,
    sampling_rate: int = 16_000,
) -> ctranslate2.StorageView:
    """Convert raw waveform → log-Mel features for CTranslate2."""
    inputs = processor(audio, return_tensors="np", sampling_rate=sampling_rate)
    return ctranslate2.StorageView.from_array(inputs.input_features)


def detect_language(
    model: ctranslate2.models.Whisper,
    features: ctranslate2.StorageView,
) -> Tuple[str, float]:
    """Return (language_token like '<|fr|>', confidence)."""
    language_token, prob = model.detect_language(features)[0][0]
    return language_token, float(prob)


# ──────────────────────────────
# Separate transcribe & translate
# ──────────────────────────────

def transcribe(
    model: ctranslate2.models.Whisper,
    features: ctranslate2.StorageView,
    processor: transformers.WhisperProcessor,
    language_token: str,
) -> str:
    """
    Transcribe audio in its original detected language.
    
    Args:
        language_token: e.g. "<|es|>", "<|fr|>" – must be the detected one
    """
    tokenizer = processor.tokenizer
    prompt = tokenizer.convert_tokens_to_ids([
        "<|startoftranscript|>",
        language_token,
        "<|transcribe|>",
        "<|notimestamps|>"
    ])

    result = model.generate(features, [prompt])
    sequence = result[0].sequences_ids[0]
    return processor.decode(sequence, skip_special_tokens=True)


def translate_to_english(
    model: ctranslate2.models.Whisper,
    features: ctranslate2.StorageView,
    processor: transformers.WhisperProcessor,
) -> str:
    """
    Translate audio directly to English (no language token needed).
    """
    tokenizer = processor.tokenizer
    prompt = tokenizer.convert_tokens_to_ids([
        "<|startoftranscript|>",
        "<|en|>",
        "<|translate|>",
        "<|notimestamps|>"
    ])

    result = model.generate(features, [prompt])
    sequence = result[0].sequences_ids[0]
    return processor.decode(sequence, skip_special_tokens=True)