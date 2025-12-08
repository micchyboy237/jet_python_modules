from __future__ import annotations

import os
from pathlib import Path
import ctranslate2
import numpy as np
import librosa
import torch
import transformers
from typing import Any, Literal, Tuple

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


from typing import Union
import numpy.typing as npt

AudioInput = Union[str, bytes, os.PathLike, npt.NDArray[Any], "torch.Tensor"]  # torch optional

def load_audio(
    audio: AudioInput,
    sr: int = 16_000,
    mono: bool = True,
) -> np.ndarray:
    """
    Load audio from a file path or directly use a numpy array / torch tensor.
    
    Returns
    -------
    np.ndarray
        Float32 array in [-1.0, 1.0], resampled to `sr` Hz, converted to mono if requested.
    """
    # Case 1: already a numpy array → normalize / resample / channel-mix
    if isinstance(audio, np.ndarray):
        y = audio.astype(np.float32, copy=False)
        current_sr = sr  # we have no way of knowing the original sr → assume target sr
    # Optional torch support (no hard dependency)
    elif hasattr(audio, "__torch__") or "torch" in str(type(audio)):
        y = audio.float().cpu().numpy()
        current_sr = sr
    # Case 2: everything else → delegate to librosa (paths, file-like objects, etc.)
    else:
        y, current_sr = librosa.load(audio, sr=None, mono=False)  # load native sr first

    # Ensure float32
    if y.dtype != np.float32:
        y = y.astype(np.float32)

    # Resample only if needed (librosa is very cheap when sr matches)
    if current_sr != sr:
        y = librosa.resample(y, orig_sr=current_sr, target_sr=sr)

    # Convert to mono if requested
    if mono and y.ndim > 1:
        y = y.mean(axis=0) if y.shape[0] <= y.shape[-1] else np.mean(y, axis=0)

    # Normalize to [-1, 1] range for integer inputs that librosa didn't already normalize
    if y.max() > 1.0 or y.min() < -1.0:
        if np.issubdtype(y.dtype, np.integer):
            bit_depth = np.iinfo(y.dtype).bits
            y = y.astype(np.float32) / (2 ** (bit_depth - 1))
        else:
            peak = np.max(np.abs(y))
            if peak > 0:
                y /= peak

    return y


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


def transcribe_audio(
    audio: Any,
) -> str:
    model_size: QuantizedModelSizes = "small"
    model_dir = Path("~/.cache/hf_ctranslate2_models").expanduser() / f"whisper-{model_size}-ct2"

    # 1. Load once
    model, processor = load_whisper_ct2_model(model_size, str(model_dir))
    audio = load_audio(audio)
    features = preprocess_audio(audio, processor)

    # 2. Detect language (optional but recommended for transcription)
    lang_token, prob = detect_language(model, features)
    print(f"Detected: {lang_token} ({prob:.2%})")

    # 3. Transcribe in original language
    text_original = transcribe(model, features, processor, language_token=lang_token)
    print("\nTranscription:")
    print(text_original)

    return text_original


def translate_to_english(
    audio: Any,
) -> str:
    model_size: QuantizedModelSizes = "small"
    model_dir = Path("~/.cache/hf_ctranslate2_models").expanduser() / f"whisper-{model_size}-ct2"

    # 1. Load once
    model, processor = load_whisper_ct2_model(model_size, str(model_dir))
    audio = load_audio(audio)
    features = preprocess_audio(audio, processor)

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