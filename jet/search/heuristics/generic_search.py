from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import (
    Generic,
    Literal,
    TypeVar,
)

T = TypeVar("T")


# ────────────────────────────────────────────────
# Types
# ────────────────────────────────────────────────


SearchLogic = Literal["AND", "OR"]
FieldWeights = dict[str, float]


@dataclass(slots=True)
class SearchResult(Generic[T]):
    item: T
    score: float
    matched_fields: list[str]
    matched_terms: dict[str, list[str]]
    highlights: dict[str, str]


# ────────────────────────────────────────────────
# Search Engine
# ────────────────────────────────────────────────


class GenericSearchEngine(Generic[T]):
    """
    Generic BM25-based in-memory search engine.

    - Works on list[T]
    - Requires text extraction function
    - Optional filter function
    - Configurable field weights
    """

    def __init__(
        self,
        items: Sequence[T],
        text_extractor: Callable[[T], dict[str, str]],
        field_weights: FieldWeights | None = None,
    ) -> None:
        self._items: list[T] = list(items)
        self._extract = text_extractor
        self._weights: FieldWeights = field_weights or {}
        self._avg_doc_length: float = 0.0

        self._prepare()

    # ────────────────────────────────────────────────
    # Index Preparation
    # ────────────────────────────────────────────────

    def _prepare(self) -> None:
        lengths: list[int] = []

        for item in self._items:
            doc = self._extract(item)
            tokens = self._tokenize(" ".join(doc.values()))
            lengths.append(len(tokens))

        self._avg_doc_length = sum(lengths) / len(lengths) if lengths else 1.0

    @staticmethod
    def _simple_stem(word: str) -> str:
        if word.endswith("ies"):
            return word[:-3] + "y"
        if word.endswith("es"):
            return word[:-2]
        if word.endswith("s"):
            return word[:-1]
        return word

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        if not text:
            return []

        words = re.findall(r"[a-z0-9#&/+-]+", text.lower())
        return [GenericSearchEngine._simple_stem(w) for w in words if len(w) >= 2]

    # ────────────────────────────────────────────────
    # Cached Index
    # ────────────────────────────────────────────────

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

    # ────────────────────────────────────────────────
    # Public Search
    # ────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        logic: SearchLogic = "AND",
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

    # ────────────────────────────────────────────────
    # Candidate Collection
    # ────────────────────────────────────────────────

    def _collect_candidates(
        self,
        terms: list[str],
        logic: SearchLogic,
    ) -> Iterable[int]:
        term_sets: list[set[int]] = []

        for term in terms:
            postings = self._index.get(term, [])
            term_sets.append({doc_id for doc_id, _ in postings})

        if not term_sets:
            return []

        if logic == "AND":
            return set.intersection(*term_sets)
        return set.union(*term_sets)

    # ────────────────────────────────────────────────
    # Scoring (BM25 Simplified)
    # ────────────────────────────────────────────────

    def _score(
        self,
        doc_id: int,
        item: T,
        terms: list[str],
    ) -> tuple[
        float,
        list[str],
        dict[str, list[str]],
        dict[str, str],
    ]:
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

                highlighted = text
                for term in terms:
                    highlighted = re.sub(
                        rf"\b{re.escape(term)}\b",
                        r"<mark>\g<0></mark>",
                        highlighted,
                        flags=re.IGNORECASE,
                    )

                highlights[field] = highlighted[:200]

        return (
            total_score,
            sorted(matched_fields),
            dict(matched_terms),
            highlights,
        )


# ────────────────────────────────────────────────
# Convenience Function
# ────────────────────────────────────────────────


def search_items(
    items: Sequence[T],
    query: str,
    *,
    text_extractor: Callable[[T], dict[str, str]],
    field_weights: FieldWeights | None = None,
    logic: SearchLogic = "AND",
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
