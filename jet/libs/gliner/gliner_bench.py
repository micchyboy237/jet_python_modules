"""
Reusable GLiNER inference benchmark with/without packing.

Features
--------
* Auto-detects Apple-silicon MPS (M1/M2) → CPU fallback.
* One-line API for packed / unpacked inference.
* CLI that prints predictions + timing + speedup.
* pytest suite with real-world examples.
"""

from __future__ import annotations

import argparse
import time
import traceback
from typing import List, Optional, Sequence, Dict, Any

import torch
from gliner import GLiNER, InferencePackingConfig


# --------------------------------------------------------------------------- #
# Helper – pick the best device for an M1 Mac
# --------------------------------------------------------------------------- #
def _best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Core wrapper
# --------------------------------------------------------------------------- #
class GLiNERWrapper:
    """
    Thin wrapper around GLiNER that hides device handling and packing config.
    """

    def __init__(
        self,
        model_name: str = "urchade/gliner_large-v2.1",
        device: str | torch.device | None = None,
        batch_size: int = 8,
        threshold: float = 0.5,
    ):
        self.batch_size = batch_size
        self.threshold = threshold
        self.device = torch.device(device) if device else _best_device()

        try:
            self.model: GLiNER = GLiNER.from_pretrained(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load GLiNER model '{model_name}'. "
                "Check internet or pass a local path."
            ) from exc

        self.model.to(self.device)
        self.model.eval()

        # Packing defaults – will be overridden per-call if needed
        self._default_max_len = getattr(self.model.config, "max_len", 512)

    # ------------------------------------------------------------------- #
    # Public API
    # ------------------------------------------------------------------- #
    def predict_no_packing(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[str]] = None,
    ) -> tuple[List[List[Dict[str, Any]]], float]:
        """Run GLiNER **without** packing."""
        if labels is None:
            labels = ["entity"]
        return self._run(packing_config=None, texts=texts, labels=labels)

    def predict_with_packing(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[str]] = None,
        *,
        max_length: int | None = None,
        streams_per_batch: int = 32,
    ) -> tuple[List[List[Dict[str, Any]]], float]:
        """
        Run GLiNER **with** inference packing.

        Parameters
        ----------
        max_length
            Override model max_len (useful to avoid truncation).
        streams_per_batch
            How many short sequences to pack into one long sequence.
        """
        if labels is None:
            labels = ["entity"]
        if max_length is None:
            max_length = self._default_max_len

        sep_id = (
            getattr(self.model.data_processor.transformer_tokenizer, "eos_token_id", None)
            or getattr(self.model.data_processor.transformer_tokenizer, "sep_token_id", None)
            or getattr(self.model.data_processor.transformer_tokenizer, "pad_token_id", None)
        )
        if sep_id is None:
            raise ValueError("Tokenizer has no known separator token.")

        packing_cfg = InferencePackingConfig(
            max_length=max_length,
            sep_token_id=sep_id,
            streams_per_batch=streams_per_batch,
        )
        return self._run(packing_config=packing_cfg, texts=texts, labels=labels)

    # ------------------------------------------------------------------- #
    # Internal runner (shared timing logic)
    # ------------------------------------------------------------------- #
    def _run(
        self,
        packing_config: InferencePackingConfig | None,
        texts: Sequence[str],
        labels: Sequence[str],
    ) -> tuple[List[List[Dict[str, Any]]], float]:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

        start = time.perf_counter()
        raw_preds = self.model.run(
            list(texts),
            list(labels),
            batch_size=self.batch_size,
            threshold=self.threshold,
            packing_config=packing_config,
        )
        elapsed = time.perf_counter() - start

        # Convert to plain Python lists (easier to compare / serialize)
        preds = [list(example) for example in raw_preds]
        return preds, elapsed


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #
def _format_entities(entities: Sequence[Dict[str, Any]]) -> str:
    if not entities:
        return "  (no entities above threshold)"
    lines = []
    for e in entities:
        txt = e["text"].replace("\n", " ")
        lines.append(
            f"  • '{txt}' | {e['label']} | score={e.get('score',0):.2f} | span={e.get('start')}:{e.get('end')}"
        )
    return "\n".join(lines)


def print_predictions(texts: Sequence[str], predictions: Sequence[Sequence[Dict[str, Any]]]) -> None:
    print("\n=== NER predictions ===")
    for i, (txt, ents) in enumerate(zip(texts, predictions), start=1):
        print(f"\nExample {i}:")
        print(txt)
        print(_format_entities(ents))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GLiNER with/without inference packing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="urchade/gliner_large-v2.1",
        help="HuggingFace model name or local folder",
    )
    parser.add_argument(
        "--device",
        choices=["mps", "cpu", "cuda"],
        help="Force device; defaults to best available on M1",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--streams-per-batch", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_cli()

    wrapper = GLiNERWrapper(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )

    # ------------------------------------------------------------------- #
    # Example data (same as the original bench script)
    # ------------------------------------------------------------------- #
    texts: List[str] = [
        "Elon Musk unveiled Tesla’s Cybertruck in Los Angeles in 2019.",
        "Apple launched the Vision Pro at WWDC 2023 in Cupertino.",
        "NASA announced that the Artemis II mission will send astronauts around the Moon in 2025.",
    ]
    # labels: List[str] = ["Person", "Organization", "Location", "Event", "Date", "Money"]
    labels = None

    # -------------------------- No packing -------------------------- #
    print("Running **without** packing …")
    preds_no, time_no = wrapper.predict_no_packing(texts, labels)
    print_predictions(texts, preds_no)
    print(f"Time (no packing): {time_no:.3f} s")

    # -------------------------- With packing -------------------------- #
    print("\nRunning **with** packing …")
    preds_pack, time_pack = wrapper.predict_with_packing(
        texts,
        labels,
        max_length=args.max_length,
        streams_per_batch=args.streams_per_batch,
    )
    print_predictions(texts, preds_pack)
    print(f"Time (packing)   : {time_pack:.3f} s")

    # -------------------------- Summary -------------------------- #
    speedup = time_no / time_pack if time_pack > 0 else float("nan")
    identical = preds_no == preds_pack
    print("\n--- Summary ---")
    print(f"Speedup          : {speedup:.2f}x")
    print(f"Identical preds  : {identical}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)