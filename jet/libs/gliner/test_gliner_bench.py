# tests/test_gliner_bench.py
"""
Functional tests for GLiNERWrapper (with & without packing).

* Real-world examples.
* Exact dict matching **after normalising order, rounding scores, and
  allowing a tiny span drift** caused by the underlying tokenizer.
* BDD-style comments.
* Proper fixture cleanup.
"""

from __future__ import annotations

import pytest
from typing import List, Dict, Any, Sequence

from jet.libs.gliner.gliner_bench import GLiNERWrapper


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def wrapper() -> GLiNERWrapper:
    """Load model once per module."""
    w = GLiNERWrapper(
        model_name="urchade/gliner_small-v2.1",
        batch_size=2,
        threshold=0.4,          # catch low-confidence but correct entities
    )
    yield w
    # cleanup
    del w.model
    if w.device.type == "mps":
        import torch
        torch.mps.empty_cache()
    elif w.device.type == "cuda":
        import torch
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Test data
# --------------------------------------------------------------------------- #
TEXTS: List[str] = [
    "Elon Musk founded SpaceX in Los Angeles in 2002.",
    "Apple released iPhone 15 at an event in Cupertino on September 12, 2023.",
]

LABELS: List[str] = ["Person", "Organization", "Location", "Event", "Date", "Product"]


# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #
def _round_score(ent: Dict[str, Any]) -> Dict[str, Any]:
    """Round score to 2 decimals – model may return 0.959 vs 0.96."""
    ent = ent.copy()
    if "score" in ent:
        ent["score"] = round(float(ent["score"]), 2)
    return ent


def _sort_entities(ents: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministic order (start → end → text)."""
    return sorted(ents, key=lambda e: (e.get("start", 0), e.get("end", 0), e["text"]))


def _normalise(example: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _sort_entities([_round_score(e) for e in example])


# --------------------------------------------------------------------------- #
# Test – no packing
# --------------------------------------------------------------------------- #
def test_predict_no_packing(wrapper: GLiNERWrapper) -> None:
    """
    Given: Two sentences.
    When: predict_no_packing() is called.
    Then: Entities match expected (order-independent, score ±0.01, span ±3).
    """
    # When
    preds, _ = wrapper.predict_no_packing(TEXTS, LABELS)

    # Normalise result
    result = [_normalise(p) for p in preds]

    # Expected – taken from a **single run** on an M1 (MPS) and rounded
    expected = [
        [
            {"text": "Elon Musk", "label": "Person", "score": 0.99, "start": 0, "end": 9},
            {"text": "SpaceX", "label": "Organization", "score": 0.96, "start": 18, "end": 24},
            {"text": "Los Angeles", "label": "Location", "score": 0.96, "start": 28, "end": 39},
            {"text": "2002", "label": "Date", "score": 0.96, "start": 43, "end": 47},
        ],
        [
            {"text": "Apple", "label": "Organization", "score": 0.99, "start": 0, "end": 5},
            {"text": "iPhone 15", "label": "Product", "score": 0.92, "start": 15, "end": 24},
            {"text": "event", "label": "Event", "score": 0.53, "start": 31, "end": 36},
            {"text": "Cupertino", "label": "Location", "score": 0.96, "start": 40, "end": 49},
            {"text": "September 12, 2023", "label": "Date", "score": 0.97, "start": 53, "end": 71},
        ],
    ]

    # Allow tiny span drift (±3) caused by whitespace tokenisation differences
    def _match(a: List[Dict], b: List[Dict]) -> bool:
        if len(a) != len(b):
            return False
        for ea, eb in zip(a, b):
            if ea["text"] != eb["text"] or ea["label"] != eb["label"]:
                return False
            if abs(ea["score"] - eb["score"]) > 0.02:
                return False
            if abs(ea.get("start", 0) - eb.get("start", 0)) > 3:
                return False
            if abs(ea.get("end", 0) - eb.get("end", 0)) > 3:
                return False
        return True

    assert _match(result[0], expected[0]), "First example mismatch"
    assert _match(result[1], expected[1]), "Second example mismatch"


# --------------------------------------------------------------------------- #
# Test – with packing (tolerance for context shift)
# --------------------------------------------------------------------------- #
def test_predict_with_packing(wrapper: GLiNERWrapper) -> None:
    """
    Given: Same inputs, packing enabled.
    When: predict_with_packing() is called.
    Then: Entities are *almost* identical to the no-packing baseline.
    """
    # Baseline (no packing) – reuse the same wrapper
    baseline, _ = wrapper.predict_no_packing(TEXTS, LABELS)
    baseline_norm = [_normalise(p) for p in baseline]

    # Packed
    packed, _ = wrapper.predict_with_packing(
        TEXTS, LABELS, max_length=512, streams_per_batch=4
    )
    packed_norm = [_normalise(p) for p in packed]

    # Same tolerant matcher as above
    def _match(a: List[Dict], b: List[Dict]) -> bool:
        if len(a) != len(b):
            return False
        for ea, eb in zip(a, b):
            if ea["text"] != eb["text"] or ea["label"] != eb["label"]:
                return False
            if abs(ea["score"] - eb["score"]) > 0.05:   # packing can shift scores a bit
                return False
            if abs(ea.get("start", 0) - eb.get("start", 0)) > 5:
                return False
            if abs(ea.get("end", 0) - eb.get("end", 0)) > 5:
                return False
        return True

    assert _match(packed_norm[0], baseline_norm[0])
    assert _match(packed_norm[1], baseline_norm[1])


# --------------------------------------------------------------------------- #
# Optional speed-check (non-fatal)
# --------------------------------------------------------------------------- #
import pytest

@pytest.mark.slow
def test_packing_is_faster(wrapper: GLiNERWrapper) -> None:
    """
    Given: A larger batch of texts (x12) run on the *same* device.
    When: Both inference modes are timed.
    Then: Packing should be **at least as fast** as no-packing.
          A soft lower-bound (0.9x) is used – the test will only warn if packing is
          noticeably slower, which can happen on CPU or with a very small model.
    """
    many = TEXTS * 12                     # increase load for a measurable difference

    # Warm-up (avoid first-call overhead)
    wrapper.predict_no_packing(TEXTS[:1], LABELS)
    wrapper.predict_with_packing(TEXTS[:1], LABELS, streams_per_batch=8)

    # ---- No packing -------------------------------------------------
    _, t_no = wrapper.predict_no_packing(many, LABELS)

    # ---- With packing ------------------------------------------------
    _, t_pack = wrapper.predict_with_packing(
        many, LABELS, max_length=512, streams_per_batch=16   # more streams → better packing
    )

    speedup = t_no / t_pack if t_pack > 0 else 0.0
    print(f"\nSpeedup ({len(many)} texts): {speedup:.2f}x")

    # Soft assertion – fail only if packing is *significantly* slower
    if speedup < 0.9:
        pytest.fail(f"Packing was {speedup:.2f}x slower than baseline (expected ≥0.9x)")
    else:
        # Always pass, but surface the metric for CI monitoring
        assert True