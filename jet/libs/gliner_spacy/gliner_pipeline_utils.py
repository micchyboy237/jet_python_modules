from __future__ import annotations
from pathlib import Path
from typing import List, Dict, TypedDict, Any, Optional
import spacy


# ---------- TypedDicts ----------

class CategoryData(TypedDict):
    """Mapping of thematic categories to their NER-related labels."""
    family: List[str]
    labor: List[str]
    education: List[str]
    movement: List[str]
    violence: List[str]


class GlinerConfig(TypedDict, total=False):
    """Configuration options for GliNER spaCy components."""
    labels: List[str]
    style: str
    threshold: float
    cat_data: CategoryData
    chunk_size: int


# ---------- Core Functions ----------

def build_label_set(cat_data: Dict[str, List[str]]) -> List[str]:
    """
    Generate a unique flat list of labels from category data.

    Args:
        cat_data: Mapping of category names to associated label terms.

    Returns:
        A unique, sorted list of labels.
    """
    labels = {label for labels in cat_data.values() for label in labels}
    return sorted(labels)


def init_gliner_pipeline(cat_data: CategoryData) -> spacy.Language:
    """
    Initialize a spaCy pipeline configured for GliNER + GliNER Cat.

    Args:
        cat_data: Mapping of category names to related label terms.

    Returns:
        A configured spaCy Language pipeline.
    """
    labels = build_label_set(cat_data)
    nlp = spacy.blank("en")

    nlp.add_pipe("sentencizer")
    nlp.add_pipe(
        "gliner_spacy",
        config={"labels": labels, "style": "span", "threshold": 0.5},
    )

    # ✅ Only include fields supported by current gliner_spacy version
    cat_config = {"cat_data": cat_data, "style": "span"}
    # Add chunk_size only if the component supports it
    try:
        nlp.add_pipe("gliner_cat", config=cat_config | {"chunk_size": 100})
    except Exception:
        nlp.add_pipe("gliner_cat", config=cat_config)

    return nlp


def load_text_from_file(file_path: str | Path) -> str:
    """
    Load text content from a UTF-8 encoded file.

    Args:
        file_path: Path to the input text file.

    Returns:
        File contents as a string.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return path.read_text(encoding="utf-8")


def process_text(nlp: spacy.Language, text: str) -> spacy.tokens.Doc:
    """
    Run the GliNER pipeline on input text.

    Args:
        nlp: The configured spaCy pipeline.
        text: Text to analyze.

    Returns:
        A spaCy Doc object containing NER and thematic annotations.
    """
    return nlp(text)


def extract_sentence_themes(doc: spacy.tokens.Doc, index: int) -> Dict[str, Any]:
    """
    Extract theme scores and span details for a specific sentence.

    Args:
        doc: The processed spaCy document.
        index: Sentence index to inspect.

    Returns:
        Dictionary with sentence text, raw scores, and contributing spans.
    """
    sents = list(doc.sents)
    if index >= len(sents):
        raise IndexError("Sentence index out of range")

    sent = sents[index]
    details = [
        {"text": ent.text, "label": ent.label_, "score": ent._.score}
        for ent in getattr(sent._, "sent_spans", [])
    ]
    return {
        "text": sent.text,
        "raw_scores": getattr(sent._, "raw_scores", {}),
        "spans": details,
    }


def visualize_doc(
    doc: spacy.tokens.Doc,
    sent_start: int = 0,
    sent_end: Optional[int] = None,
    chunk_size: int = 100,
    fig_h: int = 10,
) -> None:
    """
    Visualize GliNER Cat thematic data across sentences.

    Args:
        doc: The processed spaCy document.
        sent_start: Start index of sentences to visualize.
        sent_end: End index of sentences to visualize.
        chunk_size: Number of sentences per chunk.
        fig_h: Figure height for visualization.
    """
    doc._.visualize(
        sent_start=sent_start,
        sent_end=sent_end,
        chunk_size=chunk_size,
        fig_h=fig_h,
    )
