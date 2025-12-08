# jet/translators/utils.py
from typing import Any, Dict, List
import ctranslate2

from jet.transformers.object import make_serializable


def translation_result_to_dict(
    result: ctranslate2.TranslationResult,
) -> Dict[str, Any]:
    """
    Convert a ctranslate2.TranslationResult (pybind11 C++ object) to a plain Python dict.
    Works with current CTranslate2 ≥2.20 where hypotheses are List[List[str]].
    """
    data: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Hypotheses – always present, now a list of token lists
    # ------------------------------------------------------------------
    hypotheses: List[Dict[str, Any]] = []
    for i, token_list in enumerate(result.hypotheses):
        hyp_dict: Dict[str, Any] = {"tokens": token_list}

        # Score is optional – only added when return_scores=True
        if hasattr(result, "scores") and result.scores is not None:
            # scores is List[float], same length as hypotheses
            hyp_dict["score"] = result.scores[i]

        hypotheses.append(hyp_dict)

    data["hypotheses"] = hypotheses

    # ------------------------------------------------------------------
    # Optional top-level fields
    # ------------------------------------------------------------------
    if hasattr(result, "scores") and result.scores is not None:
        data["scores"] = result.scores

    if hasattr(result, "attention") and result.attention is not None:
        data["attention"] = result.attention

    if hasattr(result, "logits") and result.logits is not None:
        data["logits"] = result.logits

    return make_serializable(data)