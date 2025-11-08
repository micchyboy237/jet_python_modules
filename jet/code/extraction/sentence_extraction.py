import torch
import logging
from tqdm import tqdm
from wtpsplit import SaT
from typing import List, Optional, Union

from jet.wordnet.validators.sentence_validator import is_valid_sentence

# ----------------------------------------------------------------------
# DEBUG LOGGING (toggle with `debug=True`)
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Global cache for a single SaT model instance
_model_cache = {"model": None, "key": None}

def _get_model_key(model_name: str, style_or_domain: Optional[str], language: str) -> tuple:
    return (model_name, style_or_domain, language)

def _load_model(model_name: str, style_or_domain: Optional[str], language: str) -> SaT:
    model_key = _get_model_key(model_name, style_or_domain, language)
    if _model_cache["key"] != model_key:
        try:
            if style_or_domain:
                sat = SaT(model_name, style_or_domain=style_or_domain, language=language)
            else:
                sat = SaT(model_name)
            _model_cache["model"] = sat
            _model_cache["key"] = model_key
        except Exception as e:
            raise ValueError(f"Failed to load model '{model_name}' with style_or_domain '{style_or_domain}': {e}")
    return _model_cache["model"]

def group_by_empty_split(segments: List[str]) -> List[List[str]]:
    paragraphs, current = [], []
    for seg in segments:
        if seg.strip():
            current.append(seg)
        else:
            if current:
                paragraphs.append(current)
                current = []
    if current:
        paragraphs.append(current)
    return paragraphs

# ----------------------------------------------------------------------
# Joining helpers
# ----------------------------------------------------------------------
def _flatten_list(lst: List) -> List[str]:
    result: List[str] = []
    for el in lst:
        if isinstance(el, str):
            result.append(el)
        elif isinstance(el, list):
            result.extend(_flatten_list(el))
    return result

def _join_paragraph(para: Union[str, List[str]]) -> str:
    if isinstance(para, str):
        return para.strip()
    flat = _flatten_list(para)
    return "\n".join(s.strip() for s in flat if s.strip())

def strip_trailing_whitespace_after_final_newline(text: str) -> str:
    """
    Strip trailing whitespace only if it comes after the final newline.
    Preserve all line-internal trailing whitespace and the final newline.
    """
    if not text:
        return text

    # Find the last newline
    last_newline_idx = text.rfind('\n')
    if last_newline_idx == -1:
        # No newline → nothing to strip at end
        return text

    # Split: content up to and including last '\n', and trailing part
    content = text[: last_newline_idx + 1]
    trailing = text[last_newline_idx + 1 :]

    # Only strip whitespace from the trailing part
    return content + trailing.rstrip(' \t')
    
# ----------------------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------------------
def extract_sentences(
    text: Union[str, List[str]],
    model_name: str = "sat-12l-sm",
    use_gpu: bool = True,
    do_paragraph_segmentation: bool = False,
    paragraph_threshold: float = 0.5,
    style_or_domain: Optional[str] = None,
    language: str = "en",
    valid_only: bool = False,
    verbose: bool = False,
    debug: bool = False,                     # ← NEW
) -> List[str]:
    """
    Extract sentences (or paragraphs) using SaT.
    """
    if debug:
        log.setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # 1. Model loading & device
    # ------------------------------------------------------------------
    sat = _load_model(model_name, style_or_domain, language)

    device = "cpu"
    if use_gpu:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"

    if sat.device != device:
        sat.to(device)
        if device == "cuda":
            sat.half()
    log.debug(f"Model on device: {device}")

    # ------------------------------------------------------------------
    # 2. Run SaT
    # ------------------------------------------------------------------
    raw_output = sat.split(
        text,
        do_paragraph_segmentation=do_paragraph_segmentation,
        paragraph_threshold=paragraph_threshold,
        verbose=verbose,
    )

    # Always materialize generator → list
    segmented = list(raw_output)
    log.debug(f"SaT output materialized: {len(segmented)} items")
    if debug:
        # Show first few items (avoid huge logs)
        preview = str(segmented)[:500]
        log.debug(f"SaT raw preview: {preview}")

    # ------------------------------------------------------------------
    # 3. Normalise SaT output → List[List[str]]
    # ------------------------------------------------------------------
    inner_segmented = _flatten_list(segmented)
    # Check each item if it contains newline characters at the end
    # If an item does, insert empty spaces (based on number of ending newlines) after it
    processed = []
    for s in inner_segmented:
        processed.append(s)
        if s.endswith("\n"):
            newline_count = len(s) - len(s.rstrip("\n"))
            processed.extend([""] * newline_count)
    inner_segmented = processed

    # For batched input texts
    if do_paragraph_segmentation and isinstance(text, list):
        inner_segmented = _flatten_list(inner_segmented)
        # Insert empty spaces in between items
        inner_segmented = [
            s
            for i, s in enumerate(inner_segmented)
            for s in ([s, ""] if i < len(inner_segmented) - 1 else [s])
        ]
    
    # Clean out trailing whitespaces after newline
    inner_segmented = [
        strip_trailing_whitespace_after_final_newline(s)
        for s in inner_segmented
    ]

    # ------------------------------------------------------------------
    # 4. Build final paragraphs
    # ------------------------------------------------------------------
    grouped = group_by_empty_split(inner_segmented)
    
    log.debug("Flattened mixed SaT output")
    paragraphs: List[str] = [_join_paragraph(group) for group in grouped]
    sentences = paragraphs
    log.debug(f"Before validation: {len(sentences)} sentences")

    # ------------------------------------------------------------------
    # 5. Optional validation
    # ------------------------------------------------------------------
    if valid_only:
        before = len(sentences)
        sentences = [
            s for s in tqdm(sentences, desc="Filtering valid sentences")
            if is_valid_sentence(s)
        ]
        log.debug(f"Validation filtered {before - len(sentences)} sentences")

    log.debug(f"Final output length: {len(sentences)}")
    return sentences