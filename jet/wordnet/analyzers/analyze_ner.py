from dataclasses import dataclass
from typing import List, Optional, TypedDict, cast

import spacy
import torch
from jet.logger import logger
from jet.wordnet.sentence import split_sentences
from pydantic import BaseModel
from spacy import displacy
from spacy.tokens import Doc, Span, SpanGroup
from span_marker import SpanMarkerModel
from tqdm import tqdm


class SpanMarkerWord(BaseModel):
    text: str
    lemma: str
    start_idx: int
    end_idx: int
    score: float
    label: str

    def __str__(self) -> str:
        return self.text


@dataclass
class DocSentence:
    text: str
    start_char: int
    end_char: int
    token_count: int
    doc_idx: int = 0


@dataclass
class DocEntity:
    text: str
    lemma: str
    label: str
    start_char: int
    end_char: int
    score: float
    vector_norm: float | None


@dataclass
class DocNounChunk:
    text: str
    root_text: str
    root_dep: str
    root_head_text: str


@dataclass
class DocSettings:
    lang: str
    direction: str


# ── Return Type TypedDicts ──────────────────────────────────────────


class EntityDict(TypedDict):
    text: str
    lemma: str
    label: str
    start_char: int
    end_char: int
    score: float
    vector_norm: float | None


class DependencyDict(TypedDict):
    text: str
    root_text: str
    root_dep: str
    root_head_text: str


class SentenceDict(TypedDict):
    text: str
    start_char: int
    end_char: int
    token_count: int
    doc_idx: int


class SettingsDict(TypedDict):
    lang: str
    direction: str


class SpanDict(TypedDict):
    start: int
    end: int
    start_token: int
    end_token: int
    label: str
    kb_id: str
    kb_url: str


class SpansResultDict(TypedDict):
    text: str
    spans: list[SpanDict]
    title: str | None
    settings: SettingsDict
    tokens: list[str]


class AnalyzeNERResult(TypedDict):
    entities: list[EntityDict]
    dependencies: list[DependencyDict]
    sentences: list[SentenceDict]
    settings: SettingsDict
    spans: SpansResultDict


# ── Processing Functions ────────────────────────────────────────────


def process_text(
    text: str, nlp: spacy.language.Language, model: SpanMarkerModel
) -> tuple[Doc, List[SpanMarkerWord]]:
    """Process text with spaCy pipeline and SpanMarker model, returning SpanMarkerWord predictions."""
    doc = nlp(text)
    predictions = model.predict(text)
    processed_predictions = [
        SpanMarkerWord(
            text=pred["span"],
            lemma=nlp(pred["span"])[0].lemma_ if pred["span"] else "",
            start_idx=pred["char_start_index"],
            end_idx=pred["char_end_index"],
            score=pred["score"],
            label=pred["label"],
        )
        for pred in predictions
    ]
    return doc, processed_predictions


def log_entities(predictions: List[SpanMarkerWord]) -> None:
    """Log named entities with relevant details."""
    logger.newline()
    logger.debug(f"Extracted Entities ({len(predictions)}):")
    for entity in predictions:
        logger.newline()
        logger.log("Text:", entity.text, colors=["WHITE", "INFO"])
        logger.log("Lemma:", entity.lemma, colors=["WHITE", "INFO"])
        logger.log("Label:", entity.label, colors=["WHITE", "INFO"])
        logger.log("Start:", f"{entity.start_idx}", colors=["WHITE", "SUCCESS"])
        logger.log("End:", f"{entity.end_idx}", colors=["WHITE", "SUCCESS"])
        logger.log("Score:", f"{entity.score:.4f}", colors=["WHITE", "SUCCESS"])
        logger.log("---")


def log_noun_chunks(doc: Doc) -> None:
    """Log noun chunks with relevant details."""
    logger.newline()
    logger.debug(f"Extracted Noun Chunks ({len(list(doc.noun_chunks))}):")
    for chunk in doc.noun_chunks:
        logger.newline()
        logger.log("Text:", chunk.text, colors=["WHITE", "INFO"])
        logger.log("Root Text:", chunk.root.text, colors=["WHITE", "INFO"])
        logger.log("Root Dependency:", chunk.root.dep_, colors=["WHITE", "SUCCESS"])
        logger.log("Root Head Text:", chunk.root.head.text, colors=["WHITE", "SUCCESS"])
        logger.log("---")


def log_sentences(doc: Doc) -> None:
    """Log sentences with relevant details."""
    logger.newline()
    logger.debug(f"Extracted Sentences ({len(list(doc.sents))}):")
    for i, sent in enumerate(doc.sents, 1):
        logger.newline()
        logger.log(f"Sentence {i}:", sent.text, colors=["WHITE", "INFO"])
        logger.log("Start Char:", f"{sent.start_char}", colors=["WHITE", "SUCCESS"])
        logger.log("End Char:", f"{sent.end_char}", colors=["WHITE", "SUCCESS"])
        logger.log("Token Count:", f"{len(sent)}", colors=["WHITE", "SUCCESS"])
        logger.log("---")


def parse_entities(doc: Doc, predictions: List[SpanMarkerWord]) -> List[DocEntity]:
    """Parse SpanMarkerWord predictions into a list of DocEntity objects."""
    return [
        DocEntity(
            text=entity.text,
            lemma=entity.lemma,
            label=entity.label,
            start_char=entity.start_idx,
            end_char=entity.end_idx,
            score=entity.score,
            vector_norm=(
                doc[entity.start_idx : entity.end_idx].vector_norm
                if doc[entity.start_idx : entity.end_idx].has_vector
                else None
            ),
        )
        for entity in predictions
    ]


def parse_dependencies(doc: Doc) -> List[DocNounChunk]:
    """Parse a spaCy Doc into a list of DocNounChunk objects containing noun chunk details."""
    return [
        DocNounChunk(
            text=chunk.text,
            root_text=chunk.root.text,
            root_dep=chunk.root.dep_,
            root_head_text=chunk.root.head.text,
        )
        for chunk in doc.noun_chunks
    ]


def parse_sentences(doc: Doc, doc_idx: int = 0) -> List[DocSentence]:
    """Parse a spaCy Doc into a list of DocSentence objects containing sentence details."""
    return [
        DocSentence(
            text=sent.text,
            start_char=sent.start_char,
            end_char=sent.end_char,
            token_count=len(sent),
            doc_idx=doc_idx,
        )
        for sent in doc.sents
    ]


def parse_settings(doc: Doc) -> DocSettings:
    """Parse a spaCy Doc's settings into a DocSettings object."""
    return DocSettings(
        lang=doc.lang_, direction=doc.vocab.writing_system.get("direction", "ltr")
    )


def char_to_token_index(
    doc: Doc, char_start: int, char_end: int
) -> tuple[Optional[int], Optional[int]]:
    """Convert character indices to token indices in a spaCy Doc."""
    start_token = None
    end_token = None
    for token in doc:
        if token.idx <= char_start < token.idx + len(token.text):
            start_token = token.i
        if token.idx < char_end <= token.idx + len(token.text):
            end_token = token.i + 1
            break
    return start_token, end_token


def create_span_group(doc: Doc, predictions: List[SpanMarkerWord]) -> SpanGroup:
    """Create a SpanGroup from SpanMarkerWord predictions for visualization."""
    spans = []
    for entity in predictions:
        start_token, end_token = char_to_token_index(
            doc, entity.start_idx, entity.end_idx
        )
        if (
            start_token is not None
            and end_token is not None
            and start_token < len(doc)
            and end_token <= len(doc)
        ):
            try:
                span = Span(
                    doc,
                    start_token,
                    end_token,
                    label=entity.label,
                    kb_id=f"score:{entity.score:.4f}",
                )
                spans.append(span)
            except IndexError as e:
                logger.error(
                    f"Error creating span for entity '{entity.text}' "
                    f"(char {entity.start_idx}:{entity.end_idx}): {e}"
                )
        else:
            logger.warning(
                f"Skipping entity '{entity.text}' due to invalid token indices "
                f"(char {entity.start_idx}:{entity.end_idx})"
            )
    return SpanGroup(doc, name="entities", spans=spans)


def analyze_ner(texts: str | list[str]) -> AnalyzeNERResult:
    """Analyze named entities across multiple sentences with progress tracking."""

    # Force CPU to avoid MPS/Metal issues on M1 that cause Bus Errors
    device = "cpu"
    if torch.backends.mps.is_available():
        # Optional: Try MPS if you want speed, but CPU is more stable for debugging
        # device = "mps"
        pass

    nlp = spacy.load("en_core_web_sm")

    # Load model and explicitly move to CPU
    model = SpanMarkerModel.from_pretrained(
        "tomaarsen/span-marker-bert-base-fewnerd-fine-super"
    )
    model = model.to(device)
    model.eval()  # Set to evaluation mode to disable dropout/etc

    if isinstance(texts, str):
        texts = [texts]

    sentences = [
        sent for text in texts for sent in split_sentences(text, num_sentence=3)
    ]

    # Initialize accumulators with typed dicts
    all_entities: list[EntityDict] = []
    all_dependencies: list[DependencyDict] = []
    all_sentences_data: list[SentenceDict] = []
    all_spans_tokens: list[str] = []
    all_spans_span_list: list[SpanDict] = []
    settings_data: SettingsDict = {"lang": "", "direction": ""}
    spans_text: str = ""
    spans_title: Optional[str] = None
    spans_settings: SettingsDict = {"lang": "", "direction": ""}

    for idx, text in enumerate(
        tqdm(sentences, desc="Processing sentences", unit="sent")
    ):
        doc, predictions = process_text(text, nlp, model)
        doc.spans["entities"] = create_span_group(doc, predictions)

        # Accumulate entities
        all_entities.extend(
            [cast(EntityDict, e.__dict__) for e in parse_entities(doc, predictions)]
        )

        # Accumulate dependencies
        all_dependencies.extend(
            [cast(DependencyDict, d.__dict__) for d in parse_dependencies(doc)]
        )

        # Accumulate sentences with doc_idx
        sentence_dicts: list[SentenceDict] = [
            cast(SentenceDict, {**s.__dict__, "doc_idx": idx})
            for s in parse_sentences(doc, doc_idx=idx)
        ]
        all_sentences_data.extend(sentence_dicts)

        # Accumulate spans
        spans = displacy.parse_spans(doc, options={"spans_key": "entities"})
        if idx == 0:
            # Initialize with first doc's full structure
            all_spans_tokens = cast(list[str], spans.get("tokens", []))
            all_spans_span_list = cast(list[SpanDict], spans.get("spans", []))
            settings_data = cast(SettingsDict, parse_settings(doc).__dict__)
            spans_text = cast(str, spans.get("text", ""))
            spans_title = spans.get("title")
            spans_settings = cast(SettingsDict, spans.get("settings", {}))
        else:
            # Append tokens and adjust span positions for subsequent docs
            token_offset = len(all_spans_tokens)
            all_spans_tokens.extend(cast(list[str], spans.get("tokens", [])))
            for span in cast(list[SpanDict], spans.get("spans", [])):
                adjusted_span = SpanDict(
                    start=span["start"] + token_offset,
                    end=span["end"] + token_offset,
                    start_token=span["start_token"] + token_offset,
                    end_token=span["end_token"] + token_offset,
                    label=span["label"],
                    kb_id=span["kb_id"],
                    kb_url=span["kb_url"],
                )
                all_spans_span_list.append(adjusted_span)
            # Append text with space
            spans_text += " " + cast(str, spans.get("text", ""))

    # Build final spans dict
    spans_result: SpansResultDict = {
        "text": spans_text,
        "spans": all_spans_span_list,
        "title": spans_title,
        "settings": spans_settings,
        "tokens": all_spans_tokens,
    }

    logger.newline()
    logger.success(
        f"Processing complete: "
        f"{len(all_entities)} entities, "
        f"{len(all_dependencies)} dependencies, "
        f"{len(all_sentences_data)} sentences"
    )

    result: AnalyzeNERResult = {
        "entities": all_entities,
        "dependencies": all_dependencies,
        "sentences": all_sentences_data,
        "settings": settings_data,
        "spans": spans_result,
    }
    return result
