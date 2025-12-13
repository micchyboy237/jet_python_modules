import os
from typing import List, Sequence

from transformers import AutoTokenizer

from jet.translators.translator_types import (
    Device,
    BatchType,
    TranslationOptions,
    Translator,
)

# ── Device auto-detection with memory safety ──────────────────────────────────
import torch

MIN_FREE_VRAM_GB = 2.0  # Safe threshold for quantized Opus-MT models

# ── Constants ─────────────────────────────────────────────────────────────────
QUANTIZED_MODEL_PATH = os.path.expanduser("~/.cache/hf_ctranslate2_models/opus-ja-en-ct2")
DEFAULT_TOKENIZER = "Helsinki-NLP/opus-mt-ja-en"


def detect_device() -> Device:
    """
    Automatically choose the best device:
      - "cuda"  → if GPU available AND ≥ 2 GB free VRAM
      - "cpu"   → otherwise

    Uses PyTorch (already a dependency via transformers).
    """
    if not torch.cuda.is_available():
        return "cpu"

    try:
        # Get current free memory in bytes
        free_memory_bytes = torch.cuda.mem_get_info()[0]  # (free, total)
        free_memory_gb = free_memory_bytes / (1024 ** 3)
        print(f"Detected free GPU VRAM: {free_memory_gb:.2f} GB")  # Print free memory

        if free_memory_gb >= MIN_FREE_VRAM_GB:
            return "cuda"
        else:
            print(
                f"Warning: GPU has only {free_memory_gb:.2f} GB free VRAM "
                f"(< {MIN_FREE_VRAM_GB} GB), falling back to CPU"
            )
            return "cpu"
    except Exception as e:
        print(f"Warning: Could not query GPU memory ({e}), using CPU")
        return "cpu"

# ── Shared Core Translation Logic ─────────────────────────────────────────────
def _translate_core(
    source_texts: Sequence[str],
    *,
    model_path: str,
    tokenizer_name: str,
    beam_size: int,
    max_decoding_length: int,
    device: Device,
    max_batch_size: int = 0,        # 0 = let ctranslate2 decide (auto)
    batch_type: BatchType = "examples",
    **options: TranslationOptions,
) -> List[str]:
    """Internal shared implementation used by both single and batch functions."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    translator = Translator(model_path, device=device)

    # Convert all texts → list of token strings (what ctranslate2 expects)
    source_batches: list[list[str]] = [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
        for text in source_texts
    ]

    results = translator.translate_batch(
        source_batches,
        options=TranslationOptions(
            beam_size=beam_size,
            max_decoding_length=max_decoding_length,
            max_batch_size=max_batch_size,
            batch_type=batch_type,
            **options,
        ),
    )

    # Decode only the best hypothesis for each input
    translations = [
        tokenizer.decode(
            tokenizer.convert_tokens_to_ids(result.hypotheses[0])
        ).strip()
        for result in results
    ]
    return translations

# ── Public APIs (unchanged signatures) ───────────────────────────────────────
def translate_ja_to_en(
    text: str,
    *,
    model_path: str = QUANTIZED_MODEL_PATH,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    beam_size: int = 5,
    max_decoding_length: int = 512,
    device: Device | None = None,  # ← now optional!
    **options: TranslationOptions,
) -> str:
    """
    Translate a single Japanese sentence to English using a quantized Opus-MT model.

    All translation options are fully type-checked via :class:`TranslationOptions`.

    Args:
        text: Japanese input text.
        model_path: Path to the CTranslate2-converted model.
        tokenizer_name: Hugging Face tokenizer (must match the model).
        beam_size: Beam size (1 = greedy decoding).
        max_decoding_length: Maximum number of generated tokens.
        device: Device to run on ("cpu", "cuda", "auto", or ``None`` for smart auto-detect).
        **options: Any additional valid keys from :class:`TranslationOptions`
                   (e.g. ``return_scores=True``, ``sampling_temperature=0.8``).

    Returns:
        Translated English string (stripped whitespace).
    """
    if device is None or device == "auto":
        device = detect_device()

    # Single string → sequence of length 1, disable batch splitting
    result = _translate_core(
        source_texts=[text],
        model_path=model_path,
        tokenizer_name=tokenizer_name,
        beam_size=beam_size,
        max_decoding_length=max_decoding_length,
        device=device,
        max_batch_size=0,     # not needed for single example
        batch_type="examples",
        **options,
    )
    return result[0]

def batch_translate_ja_to_en(
    texts: List[str],
    *,
    model_path: str = QUANTIZED_MODEL_PATH,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    beam_size: int = 5,
    max_decoding_length: int = 512,
    device: Device | None = None,
    max_batch_size: int = 32,
    batch_type: BatchType = "examples",
    **options: TranslationOptions,
) -> List[str]:
    """
    Efficient batch translation of Japanese to English.

    Fully typed — IDEs and static checkers will catch invalid options instantly.

    Args:
        texts: List of Japanese sentences.
        model_path: Path to the CTranslate2 model.
        tokenizer_name: Matching tokenizer.
        beam_size: Beam size for decoding.
        max_decoding_length: Max tokens to generate.
        device: Runtime device ("cpu", "cuda", "auto", or ``None`` → auto-detect with memory check).
        max_batch_size: Split large inputs to reduce padding (0 = auto).
        batch_type: "examples" or "tokens".
        **options: Any additional :class:`TranslationOptions`.

    Returns:
        List of English translations in the same order.
    """
    if device is None or device == "auto":
        device = detect_device()

    return _translate_core(
        source_texts=texts,
        model_path=model_path,
        tokenizer_name=tokenizer_name,
        beam_size=beam_size,
        max_decoding_length=max_decoding_length,
        device=device,
        max_batch_size=max_batch_size,
        batch_type=batch_type,
        **options,
    )

# ── Example Usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ja_example = "おい、そんな一気に冷たいものを食べると腹を壊すぞ！"

    print("=== Single Translation ===")
    print(f"JA:  {ja_example}")
    # No device argument → automatically uses GPU if available
    en_single = translate_ja_to_en(
        ja_example,
        beam_size=5,
        return_scores=True,
        # return_attention = True,
        # return_alternatives = True,
        # return_logits_vocab=True,
    )
    print(f"EN:  {en_single}")

    ja_batch = [
        "昨日、友達と一緒に映画を見に行きました。",
        "日本は美しい国ですね！",
        "今日の天気はとても良いです。",
    ]

    print("\n=== Batch Translation ===")
    en_batch = batch_translate_ja_to_en(
        ja_batch,
        beam_size=4,
        max_batch_size=16,
        return_scores=True,
        sampling_temperature=0.9,
        replace_unknowns=True,
        # device=None → auto-detect (will use GPU on your Windows machine)
    )

    for ja, en in zip(ja_batch, en_batch):
        print(f"JA → {ja}")
        print(f"EN → {en}\n")