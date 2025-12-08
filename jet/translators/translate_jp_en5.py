from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Union, Literal, overload

import ctranslate2
from ctranslate2 import Translator, TranslationResult
from rich.console import Console
from rich.table import Table

console = Console()


class JapaneseToEnglishTranslator:
    """Lightweight wrapper around a quantized Helsinki-NLP/opus-mt-ja-en → CTranslate2 model."""

    DEFAULT_MODEL_PATH = "/Users/jethroestrada/.cache/hf_translation_models/ct2-opus-ja-en"
    DEFAULT_DEVICE = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    DEFAULT_COMPUTE_TYPE = "int8"  # matches your quantization

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: Literal["cpu", "cuda"] = DEFAULT_DEVICE,
        compute_type: Literal["default", "int8", "int16", "float16", "bfloat16", "float32"] = DEFAULT_COMPUTE_TYPE,
        intraopol_threads: int = 4,
        inter_threads: int = 1,
    ) -> None:
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"CTranslate2 model not found at {model_path}")

        self.translator = Translator(
            model_path,
            device=device,
            compute_type=compute_type,
            intra_threads=intraopol_threads,
            inter_threads=inter_threads,
        )
        self.device = device
        console.log(f"[green]Translator loaded[/] → {model_path} ({device}, {compute_type})")

    @overload
    def translate_ja_en_diverse(
        self,
        texts: str,
        *,
        max_decoding_length: int = 512,
        beam_size: int = 5,
        num_hypotheses: int = 5,
        length_penalty: float = 1.0,
        return_scores: bool = True,
        replace_unknowns: bool = True,
    ) -> List[TranslationResult]: ...

    @overload
    def translate_ja_en_diverse(
        self,
        texts: Sequence[str],
        *,
        max_decoding_length: int = 512,
        beam_size: int = 5,
        num_hypotheses: int = 5,
        length_penalty: float = 1.0,
        return_scores: bool = True,
        replace_unknowns: bool = True,
    ) -> List[List[TranslationResult]]: ...

    def translate_ja_en_diverse(
        self,
        texts: Union[str, Sequence[str]],
        *,
        max_decoding_length: int = 512,
        beam_size: int = 5,
        num_hypotheses: int = 5,
        length_penalty: float = 1.0,
        return_scores: bool = True,
        replace_unknowns: bool = True,
    ) -> Union[List[TranslationResult], List[List[TranslationResult]]]:
        """
        Translate Japanese → English with diverse beam search outputs.

        Args:
            texts: Single Japanese string or list of strings.
            max_decoding_length: Maximum generation length.
            beam_size: Beam size (higher → better quality, slower).
            num_hypotheses: Number of diverse translations to return per input (≤ beam_size).
            length_penalty: Encourage/discourage longer outputs (1.0 = neutral).
            return_scores: Include log-probability scores in results.
            replace_unknowns: Post-process to replace <unk> tokens (Marian-style).

        Returns:
            List of TranslationResult (single input) or list of lists (batch input).
        """
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            texts = list(texts)
            single_input = False

        if not texts:
            return [] if single_input else [[]]

        console.log(f"Translating {len(texts)} Japanese sentence(s) → English (beam={beam_size}, hyps={num_hypotheses})")

        results: List[List[TranslationResult]] = self.translator.translate_batch(
            source=[list(text) for text in texts],
            beam_size=beam_size,
            max_decoding_length=max_decoding_length,
            num_hypotheses=num_hypotheses,
            length_penalty=length_penalty,
            return_scores=return_scores,
            replace_unknowns=replace_unknowns,
        )

        # Return single result if input was a single string
        return results[0] if single_input else results

    def pretty_print_results(
        self,
        texts: Union[str, Sequence[str]],
        results: Union[TranslationResult, List[TranslationResult], List[List[TranslationResult]]],
    ) -> None:
        """Rich table with clean diverse translations and scores."""
        if isinstance(texts, str):
            texts = [texts]
            if isinstance(results, TranslationResult):
                results = [results]
            else:
                results = [results]  # type: ignore

        table = Table(
            title="Japanese → English (Diverse Beam Search)",
            title_style="bold magenta",
            show_lines=True,
        )
        table.add_column("Input (JA)", style="cyan", width=45)
        table.add_column("Rank", justify="center", style="dim")
        table.add_column("Translation (EN)", style="green")
        table.add_column("Score", justify="right", style="yellow")

        for ja_text, batch_results in zip(texts, results):
            # batch_results is List[TranslationResult] when num_hypotheses > 1
            for rank, res in enumerate(batch_results, start=1):
                # Modern TranslationResult object: hypotheses are already joined strings
                if isinstance(res, TranslationResult):
                    # hypotheses: List[str]  → take top hypothesis
                    translation = res.hypotheses[0] if res.hypotheses else ""
                    # scores: List[float] (one per hypothesis) → take top score
                    score_val = res.scores[0] if res.scores else None
                else:
                    # Fallback for unexpected formats (e.g., legacy dict)
                    translation = res.get("hypotheses", [""])[0] if isinstance(res, dict) else ""
                    score_val = res.get("scores", [None])[0] if isinstance(res, dict) else None

                score_str = f"{score_val:.4f}" if score_val is not None else "—"

                table.add_row(
                    ja_text if rank == 1 else "",
                    str(rank),
                    translation,
                    score_str,
                )
            table.add_row("")  # separator

        console.print(table)


def to_serializable_results(
    results: Union[TranslationResult, List[TranslationResult], List[List[TranslationResult]]],
) -> Union[dict, List[dict], List[List[dict]]]:
    """Convert TranslationResult objects to JSON-serializable dicts."""
    if isinstance(results, TranslationResult):
        return {
            "hypotheses": results.hypotheses,
            "scores": results.scores or [],
        }
    elif isinstance(results, list):
        if results and isinstance(results[0], TranslationResult):
            return [to_serializable_results(res) for res in results]
        elif results and isinstance(results[0], list):
            return [to_serializable_results(sub) for sub in results]
    return results  # already serializable


# Convenience singleton (optional)
_translator: Optional[JapaneseToEnglishTranslator] = None


def get_ja_en_translator(**kwargs) -> JapaneseToEnglishTranslator:
    """Thread-safe singleton accessor."""
    global _translator
    if _translator is None:
        _translator = JapaneseToEnglishTranslator(**kwargs)
    return _translator


# Updated convenience function (add the new helper call)
def translate_ja_en_diverse(
    texts: Union[str, Sequence[str]],
    *,
    model_path: str = JapaneseToEnglishTranslator.DEFAULT_MODEL_PATH,
    device: Literal["cpu", "cuda"] = JapaneseToEnglishTranslator.DEFAULT_DEVICE,
    compute_type: Literal["default", "int8", "int16", "float16", "bfloat16", "float32"] = "int8",
    max_decoding_length: int = 512,
    beam_size: int = 5,
    num_hypotheses: int = 5,
    length_penalty: float = 1.0,
    return_scores: bool = True,
    replace_unknowns: bool = True,
    pretty_print: bool = False,
) -> Union[List[TranslationResult], List[List[TranslationResult]]]:
    """
    One-liner reusable function for diverse Japanese → English translation.
    """
    translator = get_ja_en_translator(
        model_path=model_path,
        device=device,
        compute_type=compute_type,
    )

    results = translator.translate_ja_en_diverse(
        texts,
        max_decoding_length=max_decoding_length,
        beam_size=beam_size,
        num_hypotheses=num_hypotheses,
        length_penalty=length_penalty,
        return_scores=return_scores,
        replace_unknowns=replace_unknowns,
    )

    if pretty_print:
        translator.pretty_print_results(texts, results)

    return results

# Updated demo (use the new helper for JSON printing)
if __name__ == "__main__":
    results = translate_ja_en_diverse(
        [
            "昨日、友達と一緒に映画を見に行きました。",
            "日本は美しい国ですね！"
        ],
        beam_size=6,
        num_hypotheses=4,
        max_decoding_length=512,
        pretty_print=True,
    )
    serializable = to_serializable_results(results)
    print(f"\nResults ({len(results)}):\n{json.dumps(serializable, indent=2)}")