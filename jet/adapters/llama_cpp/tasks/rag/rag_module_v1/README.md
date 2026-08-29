# RAG Module v1

Eval-driven knowledge search tool for ReAct agents.

## Quick Start

```python
from jet.adapters.llama_cpp.tasks.rag.rag_module_v1 import search_knowledge

result = search_knowledge("What is the domestic flight reimbursement limit?")
print(result["status"])       # "found" | "abstained" | "error"
print(result["answer_context"])
print(result["sources"])      # List of {chunk_id, doc_title, relevance_score, arms}
```

## Run Evals

```bash
cd jet_python_modules
python -m jet.adapters.llama_cpp.tasks.rag.rag_module_v1.eval_rag \
    --dataset jet/adapters/llama_cpp/tasks/rag/rag_module_v1/data/eval_v1.jsonl
```

Target metrics (MVP):

- Recall@5 ≥ 0.85 (non-abstain examples)
- Abstention accuracy ≥ 0.95
- Parse rate = 1.0
- p95 latency < 800ms

## Architecture

```
query → validate → rewrite → extract metadata → filter corpus
  → vector retrieve + BM25 retrieve → RRF fusion → rerank
  → dynamic threshold → format context (token-aware) → structured output
```

## Files

| File                   | Purpose                                   |
| ---------------------- | ----------------------------------------- |
| `search_knowledge.py`  | Main orchestrator + public API            |
| `retrieval.py`         | Vector/BM25/RRF/rerank adapters           |
| `query_processing.py`  | Query rewrite + metadata extraction       |
| `formatting.py`        | Token-aware context assembly              |
| `corpus.py`            | Corpus loading + metadata filtering       |
| `schemas.py`           | Chunk, RetrievedChunk, SearchResult types |
| `config.py`            | Frozen RAGConfig dataclass                |
| `eval_rag.py`          | Local eval runner (<60s)                  |
| `data/corpus_v1.jsonl` | Chunk corpus with metadata                |
| `data/eval_v1.jsonl`   | 40-example golden test set                |

## Key Design Decisions

- **In-memory retrieval for eval phase.** BM25 uses `jet.vectors.reranker.bm25` over full corpus. Indexed retrieval deferred to production.
- **Metadata extraction is hybrid.** Deterministic keyword hints first, optional LLM fallback via structured output.
- **Abstention over hallucination.** Dynamic threshold (mean + 0.5σ) with zero-variance guard. Returns explicit `ABSTAINED` status.
- **Token-aware formatting.** Uses `token_utils.count_tokens()` to enforce `max_context_tokens` budget.
- **Arm attribution preserved.** RRF fusion tracks `arms: ["vector", "bm25"]` per result for observability.

## Adding Context to Coreference Examples

Eval rows with ambiguous queries (`"compare them"`, `"the one from last month"`) require `thought_context`:

```jsonl
{
  "id": "failure_007",
  "query": "compare them",
  "thought_context": "User previously asked about Q3 and Q4 financial summaries.",
  "expected_chunk_ids": [
    "fin_q3_2025#summary",
    "fin_q4_2025#summary"
  ],
  "should_abstain": false,
  "metadata_filter": {
    "doc_type": "financial_report"
  }
}
```

Update `eval_rag.py` to pass it:

```python
pred = search_knowledge(ex["query"], thought_context=ex.get("thought_context", ""))
```

````

#### `config.py`
```python
"""Frozen RAG configuration with validated defaults."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RAGConfig:
    # Retrieval
    vector_top_k: int = 20
    bm25_top_k: int = 20
    fusion_top_k: int = 20
    rerank_top_n: int = 10

    # Per-arm thresholds (applied BEFORE fusion)
    vector_min_score: float | None = None
    bm25_min_score: float = 0.01

    # Abstention
    default_abstention_threshold: float = 0.55
    min_absolute_threshold: float = 0.50
    zero_variance_margin: float = 0.05

    # Output
    max_context_tokens: int = 2000
    max_query_chars: int = 1000
    max_thought_context_chars: int = 4000

    # Feature flags
    enable_query_rewrite: bool = True
    enable_metadata_extraction: bool = True
    enable_rerank: bool = True
````

#### `schemas.py`

```python
"""Core data types for RAG module v1."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SearchStatus(str, Enum):
    FOUND = "found"
    ABSTAINED = "abstained"
    ERROR = "error"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    arms: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    status: SearchStatus
    answer_context: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    query_used: str = ""
    metadata_applied: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False
    _latency_ms: int = 0

    def to_dict(self, include_internal: bool = True) -> dict:
        d = {
            "status": self.status.value,
            "answer_context": self.answer_context,
            "sources": self.sources,
            "query_used": self.query_used,
            "metadata_applied": self.metadata_applied,
            "truncated": self.truncated,
        }
        if include_internal:
            d["_latency_ms"] = self._latency_ms
        return d
```

#### `corpus.py`

```python
"""Corpus loading and metadata filtering."""

import json
from pathlib import Path
from typing import Any

from .schemas import Chunk


def load_corpus(path: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chunks.append(Chunk(
                chunk_id=row["chunk_id"],
                doc_id=row.get("doc_id", row["chunk_id"].split("#")[0]),
                doc_title=row.get("doc_title", ""),
                content=row["content"],
                metadata=row.get("metadata", {}),
            ))
    return chunks


def metadata_matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if expected is None:
            continue
        actual = metadata.get(key)
        if key.endswith("_gte"):
            base = key.removesuffix("_gte")
            if metadata.get(base) is None or metadata[base] < expected:
                return False
        elif key.endswith("_lt"):
            base = key.removesuffix("_lt")
            if metadata.get(base) is None or metadata[base] >= expected:
                return False
        else:
            if actual != expected:
                return False
    return True


def filter_chunks(chunks: list[Chunk], filters: dict[str, Any]) -> list[Chunk]:
    if not filters:
        return chunks
    return [c for c in chunks if metadata_matches(c.metadata, filters)]
```

#### `query_processing.py`

```python
"""Query validation, rewriting, and metadata extraction."""

import re
import unicodedata
from typing import Any

from pydantic import BaseModel, Field

from jet.adapters.llama_cpp.llm_utils import chat

CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

DOC_TYPE_HINTS = {
    "travel": "hr_policy", "vacation": "hr_policy",
    "parental leave": "hr_policy", "remote work": "hr_policy",
    "security incident": "security_policy", "security": "security_policy",
    "expense": "finance_policy", "revenue": "financial_report",
    "financial": "financial_report", "vpn": "it_faq",
    "software license": "it_procedure", "email attachment": "it_policy",
    "all-hands": "calendar", "performance review": "calendar",
    "payroll": "directory",
}


class QueryRewriteResult(BaseModel):
    rewritten_query: str = Field(description="Self-contained search query")


class MetadataFilters(BaseModel):
    doc_type: str | None = None
    region: str | None = None
    date_gte: str | None = None
    date_lt: str | None = None


def normalize_input(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return CONTROL_CHAR_RE.sub("", text).strip()


def validate_query(query: str, max_chars: int) -> str:
    query = normalize_input(query)
    if not query:
        raise ValueError("Query must not be empty")
    if len(query) > max_chars:
        raise ValueError(f"Query too long: {len(query)} > {max_chars}")
    return query


def rewrite_query(query: str, thought_context: str = "") -> str:
    prompt = (
        f"Rewrite this agent query into one self-contained search statement.\n"
        f"Rules: Resolve pronouns using context. Do not invent entities. "
        f"Remove filler. Return exactly one concise query.\n\n"
        f"Query: {query}\nContext: {thought_context}"
    )
    result = chat(prompt, temperature=0.0, max_tokens=120,
                  response_format=QueryRewriteResult,
                  project_name="rag-query-rewrite", capture_content=False)
    s = getattr(result, "structured", None)
    if s and s.success and s.parsed:
        return s.parsed.rewritten_query.strip() or query
    return query


def extract_metadata(query: str, use_llm: bool = False) -> dict[str, Any]:
    q = query.lower()
    filters: dict[str, Any] = {}
    for phrase, dt in DOC_TYPE_HINTS.items():
        if phrase in q:
            filters["doc_type"] = dt
            break
    if "apac" in q:
        filters["region"] = "APAC"
    elif "emea" in q:
        filters["region"] = "EMEA"
    if "last month" in q:
        filters.update({"date_gte": "2026-07-01", "date_lt": "2026-08-01"})
    elif "last week" in q:
        filters["date_gte"] = "2026-08-15"

    if not use_llm:
        return filters

    prompt = (f"Extract metadata filters from this query. Allowed: doc_type, region, "
              f"date_gte, date_lt. Return null for unknown.\nQuery: {query}")
    result = chat(prompt, temperature=0.0, max_tokens=150,
                  response_format=MetadataFilters,
                  project_name="rag-metadata-extraction", capture_content=False)
    s = getattr(result, "structured", None)
    if s and s.success and s.parsed:
        return {**filters, **s.parsed.model_dump(exclude_none=True)}
    return filters
```

#### `retrieval.py`

```python
"""Vector, BM25, RRF fusion, and reranking adapters."""

from collections import defaultdict

from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.scoring_utils import cosine_similarity
from jet.adapters.llama_cpp.rerank_utils import rerank
from jet.vectors.reranker.bm25 import rerank_bm25

from .schemas import Chunk, RetrievedChunk


def vector_retrieve(query: str, chunks: list[Chunk], top_k: int,
                    min_score: float | None = None) -> list[RetrievedChunk]:
    if not chunks:
        return []
    texts = [c.content for c in chunks]
    q_emb = embed(query)
    d_embs = embed(texts, show_progress=False)
    results = []
    for chunk, d_emb in zip(chunks, d_embs):
        score = float(cosine_similarity(q_emb, d_emb))
        if min_score is None or score >= min_score:
            results.append(RetrievedChunk(chunk=chunk, score=score,
                                          vector_score=score, arms=["vector"]))
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def bm25_retrieve(query: str, chunks: list[Chunk], top_k: int,
                  min_score: float = 0.0) -> list[RetrievedChunk]:
    if not chunks:
        return []
    documents = [c.content for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metadatas = [c.metadata for c in chunks]
    _, raw = rerank_bm25(query=query, documents=documents, ids=ids, metadatas=metadatas)
    by_id = {c.chunk_id: c for c in chunks}
    results = []
    for r in raw:
        score = float(r["score"])
        if score < min_score:
            continue
        results.append(RetrievedChunk(chunk=by_id[r["id"]], score=score,
                                      bm25_score=score, arms=["bm25"]))
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def rrf_fusion(vector_results: list[RetrievedChunk],
               bm25_results: list[RetrievedChunk],
               *, k: int = 60, top_k: int = 20) -> list[RetrievedChunk]:
    fused: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = defaultdict(float)
    for rank, r in enumerate(vector_results, 1):
        cid = r.chunk.chunk_id
        scores[cid] += 1.0 / (k + rank)
        if cid not in fused:
            fused[cid] = r
        else:
            fused[cid].vector_score = r.vector_score
            fused[cid].arms = sorted(set(fused[cid].arms + ["vector"]))
    for rank, r in enumerate(bm25_results, 1):
        cid = r.chunk.chunk_id
        scores[cid] += 1.0 / (k + rank)
        if cid not in fused:
            fused[cid] = r
        else:
            fused[cid].bm25_score = r.bm25_score
            fused[cid].arms = sorted(set(fused[cid].arms + ["bm25"]))
    for cid, r in fused.items():
        r.score = scores[cid]
    out = list(fused.values())
    out.sort(key=lambda r: r.score, reverse=True)
    return out[:top_k]


def rerank_chunks(query: str, candidates: list[RetrievedChunk],
                  top_n: int) -> list[RetrievedChunk]:
    if not candidates:
        return []
    docs = [r.chunk.content for r in candidates]
    rr = rerank(query=query, documents=docs,
                top_n=min(top_n, len(docs)), method="auto", normalize_scores=True)
    out = []
    for item in rr:
        orig = candidates[item["index"]]
        orig.rerank_score = float(item["score"])
        orig.score = float(item["score"])
        out.append(orig)
    out.sort(key=lambda r: r.score, reverse=True)
    return out
```

#### `formatting.py`

```python
"""Token-aware context assembly."""

from jet.adapters.llama_cpp.token_utils import count_tokens
from .schemas import RetrievedChunk


def format_context(results: list[RetrievedChunk],
                   max_tokens: int) -> tuple[str, bool]:
    parts: list[str] = []
    truncated = False
    for r in results:
        block = f"[Source: {r.chunk.doc_title} | {r.chunk.chunk_id}]\n{r.chunk.content.strip()}\n"
        candidate = "\n\n".join(parts + [block])
        if count_tokens(candidate) <= max_tokens:
            parts.append(block)
        else:
            truncated = True
            break
    return "\n\n".join(parts).strip(), truncated
```

#### `search_knowledge.py`

_(Same as previous response — see full implementation above. No changes needed.)_

#### `eval_rag.py`

_(Same corrected version as previous response — fixes unhashable dict bug, adds thought_context support, cleaner metric names.)_

#### `data/corpus_v1.jsonl`

Create this file with chunks matching all `expected_chunk_ids` in `eval_v1.jsonl`. Minimum viable entry:

```jsonl
{
  "chunk_id": "policy_travel_v3#sec4.2",
  "doc_id": "policy_travel_v3",
  "doc_title": "Travel Policy v3",
  "content": "Domestic flights are reimbursed up to $800 per trip. Receipts required for amounts over $75.",
  "metadata": {
    "doc_type": "hr_policy",
    "version": "v3",
    "date": "2026-01-15"
  }
}
```

Generate entries for all 40 eval examples before running evals.

---

### 3. Verification Checklist Before First Eval Run

- [ ] `corpus_v1.jsonl` contains all chunk IDs referenced in `eval_v1.jsonl`
- [ ] `eval_v1.jsonl` coreference examples have `thought_context` field
- [ ] `python -c "from jet.adapters.llama_cpp.tasks.rag.rag_module_v1 import search_knowledge"` succeeds
- [ ] `parse_rate = 1.0` on first run (fix schema issues before tuning retrieval)
- [ ] `_latency_ms` appears in eval output
- [ ] Failures print predicted vs expected chunk IDs
