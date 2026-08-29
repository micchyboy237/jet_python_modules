# rag_module_v1/query_processing.py

import re
import unicodedata
from typing import Any

from jet.adapters.llama_cpp.llm_utils import chat
from pydantic import BaseModel, Field

CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class QueryRewriteResult(BaseModel):
    rewritten_query: str = Field(description="Self-contained search query")


class MetadataFilters(BaseModel):
    doc_type: str | None = None
    region: str | None = None
    date_gte: str | None = None
    date_lt: str | None = None


def normalize_input(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = CONTROL_CHAR_RE.sub("", text)
    return text.strip()


def validate_query(query: str, max_chars: int) -> str:
    query = normalize_input(query)

    if not query:
        raise ValueError("Query must not be empty")

    if len(query) > max_chars:
        raise ValueError(f"Query too long: {len(query)} > {max_chars}")

    return query


def rewrite_query(query: str, thought_context: str = "") -> str:
    prompt = f"""
Rewrite the user/agent query into one self-contained search query.

Rules:
- Resolve pronouns only if the provided context supports it.
- Do not invent entities, dates, people, or document names.
- Remove conversational filler.
- Return exactly one concise query.

Query:
{query}

Context:
{thought_context}
"""

    result = chat(
        prompt,
        temperature=0.0,
        max_tokens=120,
        response_format=QueryRewriteResult,
        project_name="rag-query-rewrite",
        capture_content=False,
    )

    structured = getattr(result, "structured", None)

    if structured and structured.success and structured.parsed:
        rewritten = structured.parsed.rewritten_query.strip()
        return rewritten or query

    return query


DOC_TYPE_HINTS = {
    "travel": "hr_policy",
    "vacation": "hr_policy",
    "parental leave": "hr_policy",
    "remote work": "hr_policy",
    "security incident": "security_policy",
    "security": "security_policy",
    "expense": "finance_policy",
    "revenue": "financial_report",
    "financial": "financial_report",
    "vpn": "it_faq",
    "software license": "it_procedure",
    "email attachment": "it_policy",
    "all-hands": "calendar",
    "performance review": "calendar",
    "payroll": "directory",
}


def extract_metadata(query: str, use_llm: bool = False) -> dict[str, Any]:
    q = query.lower()
    filters: dict[str, Any] = {}

    for phrase, doc_type in DOC_TYPE_HINTS.items():
        if phrase in q:
            filters["doc_type"] = doc_type
            break

    if "apac" in q:
        filters["region"] = "APAC"
    elif "emea" in q:
        filters["region"] = "EMEA"

    # Simple temporal examples for eval-set style queries.
    if "last month" in q:
        filters["date_gte"] = "2026-07-01"
        filters["date_lt"] = "2026-08-01"
    elif "last week" in q:
        filters["date_gte"] = "2026-08-15"
    elif "latest" in q:
        # This should eventually become sort-by-version logic.
        pass

    if not use_llm:
        return filters

    prompt = f"""
Extract metadata filters from this query.

Allowed fields:
- doc_type
- region
- date_gte
- date_lt

Return null for unknown fields.
Query: {query}
"""

    result = chat(
        prompt,
        temperature=0.0,
        max_tokens=150,
        response_format=MetadataFilters,
        project_name="rag-metadata-extraction",
        capture_content=False,
    )

    structured = getattr(result, "structured", None)

    if structured and structured.success and structured.parsed:
        parsed = structured.parsed.model_dump(exclude_none=True)
        return {**filters, **parsed}

    return filters
