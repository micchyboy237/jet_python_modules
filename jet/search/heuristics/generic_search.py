# jet_python_modules/jet/search/heuristics/generic_search.py
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, Literal, TypeVar

T = TypeVar("T")
SearchLogic = Literal["AND", "OR"]
FieldWeights = dict[str, float]


@dataclass(slots=True)
class SearchResult(Generic[T]):
    item: T
    score: float
    matched_fields: list[str]
    matched_terms: dict[str, list[str]]
    highlights: dict[str, str]

    def to_dict(self) -> dict:
        """
        Convert SearchResult to JSON-serializable dictionary.
        Assumes item is already JSON-serializable.
        """
        return {
            "item": self.item,
            "score": self.score,
            "matched_fields": self.matched_fields,
            "matched_terms": self.matched_terms,
            "highlights": self.highlights,
        }


class GenericSearchEngine(Generic[T]):
    """
    Generic BM25-based in-memory search engine (English-focused).
    - Works on list[T]
    - Requires text extraction function
    - Optional filter function
    - Configurable field weights
    """

    def __init__(
        self,
        items: Sequence[T],
        text_extractor: Callable[[T], dict[str, str]] | None = None,
        field_weights: FieldWeights | None = None,
    ) -> None:
        self._items: list[T] = list(items)
        self._extract: Callable[[T], dict[str, str]] = (
            text_extractor or self._default_extract
        )
        self._weights: FieldWeights = field_weights or {}
        self._avg_doc_length: float = 0.0
        self._prepare()

    def _default_extract(self, item: T) -> dict[str, str]:
        """
        Default extraction logic:

        - If item is dict (including TypedDict), extract all string fields.

        - Otherwise, require explicit extractor.
        """
        if isinstance(item, dict):
            return {
                str(key): value for key, value in item.items() if isinstance(value, str)
            }

        raise TypeError("text_extractor must be provided for non-dict items.")

    def _prepare(self) -> None:
        lengths: list[int] = []
        for item in self._items:
            doc = self._extract(item)
            tokens = self._tokenize(" ".join(doc.values()))
            lengths.append(len(tokens))
        self._avg_doc_length = sum(lengths) / len(lengths) if lengths else 1.0

    @staticmethod
    def _simple_stem(word: str) -> str:
        # Basic English stemming rules
        if word.endswith("ies"):
            return word[:-3] + "y"
        if word.endswith("es"):
            return word[:-2]
        if word.endswith("s"):
            return word[:-1]
        if word.endswith("ing"):
            return word[:-3] + "e" if len(word) > 5 else word[:-3]
        if word.endswith("ed"):
            return word[:-2]
        return word

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        if not text:
            return []

        # English-focused pattern:
        # - Words (a-z + digits, allowing internal . _ -)
        # - Floating point numbers & scientific notation
        # - Hex literals
        pattern = (
            r"[a-z0-9]+(?:[._-][a-z0-9]+)*"  # words, versions, identifiers
            r"|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"  # decimals + scientific notation
            r"|\b0x[a-fA-F0-9]+\b"  # hex
        )

        words = re.findall(pattern, text.lower())
        return [GenericSearchEngine._simple_stem(w) for w in words if len(w) >= 2]

    @cached_property
    def _index(self) -> dict[str, list[tuple[int, str]]]:
        index: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for idx, item in enumerate(self._items):
            doc = self._extract(item)
            for field_name, text in doc.items():
                for token in self._tokenize(text):
                    index[token].append((idx, field_name))
        return index

    @cached_property
    def _doc_freq(self) -> dict[str, int]:
        df: dict[str, int] = {}
        for term, postings in self._index.items():
            df[term] = len({doc_id for doc_id, _ in postings})
        return df

    def search(
        self,
        query: str,
        *,
        logic: SearchLogic = "OR",
        limit: int = 20,
        offset: int = 0,
        filter_fn: Callable[[T], bool] | None = None,
    ) -> list[SearchResult[T]]:
        if not query.strip():
            return []

        terms = self._tokenize(query)
        if not terms:
            return []

        candidate_ids = self._collect_candidates(terms, logic)

        results: list[SearchResult[T]] = []
        for idx in candidate_ids:
            item = self._items[idx]
            if filter_fn and not filter_fn(item):
                continue

            score, fields, matched, highlights = self._score(idx, item, terms)
            if score > 0:
                results.append(
                    SearchResult(
                        item=item,
                        score=round(score, 4),
                        matched_fields=fields,
                        matched_terms=matched,
                        highlights=highlights,
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[offset : offset + limit]

    def _collect_candidates(
        self,
        terms: list[str],
        logic: SearchLogic,
    ) -> set[int]:
        if not terms:
            return set()

        term_sets: list[set[int]] = []
        for term in terms:
            postings = self._index.get(term, [])
            doc_ids = {doc_id for doc_id, _ in postings}
            term_sets.append(doc_ids)

        if logic == "AND":
            if len(term_sets) == 1:
                return term_sets[0]
            return set.intersection(*term_sets) if term_sets else set()

        # OR
        return set.union(*term_sets) if term_sets else set()

    def _score(
        self,
        doc_id: int,
        item: T,
        terms: list[str],
    ) -> tuple[float, list[str], dict[str, list[str]], dict[str, str]]:
        doc = self._extract(item)
        N = len(self._items)
        avgdl = self._avg_doc_length or 1.0
        total_score = 0.0
        matched_fields: set[str] = set()
        matched_terms: dict[str, list[str]] = defaultdict(list)
        highlights: dict[str, str] = {}

        for field, text in doc.items():
            tokens = self._tokenize(text)
            tf = Counter(tokens)
            field_score = 0.0
            field_weight = self._weights.get(field, 1.0)

            for term in terms:
                if term not in tf:
                    continue

                df = self._doc_freq.get(term, 1)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                term_freq = tf[term]

                # BM25 saturation + length normalization (k1=1.5, b=0.75)
                norm = (
                    term_freq
                    * (1.5 + 1)
                    / (term_freq + 1.5 * (1 - 0.75 + 0.75 * len(tokens) / avgdl))
                )

                field_score += idf * norm * field_weight
                matched_fields.add(field)
                matched_terms[term].append(field)

            if field_score > 0:
                total_score += field_score

                # Highlighting (case-insensitive, preserve original case)
                highlighted = text
                for term in terms:
                    highlighted = re.sub(
                        rf"\b{re.escape(term)}\b",
                        r"<mark>\g<0></mark>",
                        highlighted,
                        flags=re.IGNORECASE,
                    )
                highlights[field] = highlighted[:200] + (
                    "..." if len(highlighted) > 200 else ""
                )

        return (
            total_score,
            sorted(matched_fields),
            dict(matched_terms),
            highlights,
        )


def search_items(
    items: Sequence[T],
    query: str,
    *,
    text_extractor: Callable[[T], dict[str, str]] | None = None,
    field_weights: FieldWeights | None = None,
    logic: SearchLogic = "OR",
    limit: int = 20,
    offset: int = 0,
    filter_fn: Callable[[T], bool] | None = None,
) -> list[SearchResult[T]]:
    """
    Stateless helper for one-off searches.
    """
    engine = GenericSearchEngine(
        items=items,
        text_extractor=text_extractor,
        field_weights=field_weights,
    )
    return engine.search(
        query=query,
        logic=logic,
        limit=limit,
        offset=offset,
        filter_fn=filter_fn,
    )
