"""
Fast Japanese → English translator using CTranslate2-converted Opus-MT model.

Install once:
    pip install transformers sentencepiece ctranslate2

Convert model (run once):
    ct2-transformers-converter --model Helsinki-NLP/opus-mt-ja-en --output_dir ~/.cache/hf_translation_models/ct2-opus-ja-en
"""

import os
from typing import List

import ctranslate2
from transformers import AutoTokenizer

DEFAULT_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-ja-en"
DEFAULT_CT2_MODEL_DIR = os.path.expanduser("~/.cache/hf_translation_models/ct2-opus-ja-en")


class FastOpusMT:
    def __init__(
        self,
        model: str = DEFAULT_TRANSLATION_MODEL,
        ct2_model_dir: str = DEFAULT_CT2_MODEL_DIR,
        device: str = "cpu",
        max_decoding_length: int = 512,
    ) -> None:
        """
        Fast translator using a CTranslate2-converted Opus-MT model.

        Args:
            model: Original HF model name (only used for tokenizer)
            ct2_model_dir: Path to the converted CTranslate2 model directory
            device: "cpu" or "cuda"
            max_decoding_length: Maximum tokens to generate
        """
        if not os.path.isdir(ct2_model_dir):
            raise FileNotFoundError(
                f"CTranslate2 model not found at {ct2_model_dir}\n"
                "Run: ct2-transformers-converter --model Helsinki-NLP/opus-mt-ja-en --output_dir ~/.cache/hf_translation_models/ct2-opus-ja-en"
            )

        self.translator = ctranslate2.Translator(ct2_model_dir, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_decoding_length = max_decoding_length

    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translate a batch of Japanese → English.
        Correctly handles tokenization and proper decoding of CTranslate2 output.
        """
        # 1. Tokenize Japanese text → list of subword tokens (strings)
        source_tokens: List[List[str]] = [
            self.tokenizer.tokenize(text) for text in texts
        ]

        # 2. Run translation with CTranslate2
        results = self.translator.translate_batch(
            source_tokens,
            beam_size=5,                     # good quality/speed trade-off
            max_decoding_length=self.max_decoding_length,
            num_hypotheses=1,
        )

        # 3. Convert the generated token **IDs** (not strings) back to text
        translations: List[str] = []
        for result in results:
            # CTranslate2 returns tokens, not IDs → convert first
            tokens = result.hypotheses[0]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            text = self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
            )

            # Marian/SentencePiece models sometimes output "▁" as literal space marker
            # → clean it up once and for all
            text = text.replace("▁", " ").strip()

            translations.append(text)

        return translations

    def translate(self, text: str) -> str:
        """Translate a single Japanese string to English."""
        return self.translate_batch([text])[0]


if __name__ == "__main__":
    fast = FastOpusMT(device="cpu")  # use "cuda" if available and built with CUDA support

    samples = [
        "今日は忙しいですが、夕食は一緒にどうですか？",
        # "翻訳テスト。短い文と長い文の両方を試します。このモデルはCTranslate2で高速化されており、バッチ処理にも対応しています。",
    ]

    print("Translating...\n")
    outs = fast.translate_batch(samples)

    for ja, en in zip(samples, outs):
        print("JA:", ja)
        print("EN:", en)
        print("---\n")
