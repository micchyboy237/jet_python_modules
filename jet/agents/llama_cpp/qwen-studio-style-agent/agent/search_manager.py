import json
import logging

from jet.adapters.llama_cpp.llm_utils import chat
from jet.libs.llama_cpp.usage.chat_stream_types import StreamCompletionResult
from tools.conflict_detector import detect_conflicts
from tools.web_extractor import web_extractor
from tools.web_search import web_search

from agent.config import Config

logger = logging.getLogger(__name__)

SEARCH_MANAGER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_manager",
        "description": (
            "Managed search with automatic conflict detection and enforced verification. "
            "Use INSTEAD of raw web_search for any factual query requiring accuracy. "
            "Automatically extracts and cross-references when snippets contain conflicting data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The factual question to research",
                },
                "verification_required": {
                    "type": "boolean",
                    "description": "Force extraction even if no conflicts detected (default: auto-detect)",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    },
}


def search_manager(query: str, verification_required: bool = False) -> str:
    """
    Enforced verification search pipeline:
    1. Dispatch: web_search for candidates
    2. Reflect: detect conflicts in snippets
    3. Verify: MANDATORY extraction if conflicts found
    4. Aggregate: cross-reference and synthesize verified facts
    """
    logger.info(f"🔍 SearchManager starting for: {query[:80]}")

    # === STAGE 1: DISPATCH ===
    search_results = web_search(query=query, num_results=5)
    if search_results.startswith("No relevant") or search_results.startswith(
        "Search error"
    ):
        return search_results

    # === STAGE 2: REFLECT (Conflict Detection) ===
    conflict_report = detect_conflicts(search_results)
    logger.info(f"🔎 Conflict analysis: {conflict_report['reason']}")

    needs_verification = verification_required or conflict_report["has_conflict"]

    if not needs_verification:
        # No conflicts — safe to proceed with snippet-based answer
        logger.info("✅ No conflicts detected; proceeding with snippet synthesis")
        return _synthesize_with_llm(query, search_results, conflict_report)

    # === STAGE 3: VERIFY (Mandatory Extraction) ===
    logger.warning(
        "⚠️ Conflicts detected or verification forced; initiating mandatory extraction"
    )

    # Extract URLs from search results for verification
    urls_to_verify = _extract_top_urls(search_results, max_urls=2)

    if not urls_to_verify:
        logger.warning(
            "Could not extract URLs for verification; falling back to snippets"
        )
        return _synthesize_with_llm(query, search_results, conflict_report)

    extraction_results = []
    for i, url in enumerate(urls_to_verify):
        goal = f"Verify the exact answer to: {query}. Extract precise values with source context."
        logger.info(
            f"📄 Extracting from URL {i + 1}/{len(urls_to_verify)}: {url[:60]}..."
        )
        extracted = web_extractor(url=url, goal=goal)
        extraction_results.append({"url": url, "content": extracted})

    # === STAGE 4: AGGREGATE & CROSS-REFERENCE ===
    return _cross_reference_and_synthesize(
        query=query,
        search_snippets=search_results,
        extractions=extraction_results,
        conflict_report=conflict_report,
    )


def _extract_top_urls(search_results: str, max_urls: int = 2) -> list[str]:
    """Parse URLs from formatted search results."""
    urls = []
    for line in search_results.split("\n"):
        if line.strip().startswith("URL:"):
            url = line.replace("URL:", "").strip()
            if url.startswith("http"):
                urls.append(url)
                if len(urls) >= max_urls:
                    break
    return urls


def _synthesize_with_llm(query: str, search_results: str, conflict_report: dict) -> str:
    """Single-turn synthesis when no verification needed."""
    result: StreamCompletionResult = chat(
        prompt_or_messages=[
            {
                "role": "user",
                "content": (
                    f"Answer this question using ONLY the provided search results.\n"
                    f"Question: {query}\n\n"
                    f"Search Results:\n{search_results}\n\n"
                    f"Conflict Analysis: {json.dumps(conflict_report)}\n\n"
                    f"If the answer is clear from snippets, state it with source. "
                    f"If uncertain, say so explicitly."
                ),
            }
        ],
        model=Config.LLAMA_MODEL,
        temperature=0.0,
        max_tokens=512,
        project_name="qwen-studio-search-manager",
        phoenix_url=Config.PHOENIX_URL,
    )
    return result.content.strip()


def _cross_reference_and_synthesize(
    query: str,
    search_snippets: str,
    extractions: list[dict[str, str]],
    conflict_report: dict,
) -> str:
    """Cross-reference extracted content and produce verified answer."""
    extraction_summary = "\n\n".join(
        f"[Source {i + 1}: {e['url'][:80]}]\n{e['content']}"
        for i, e in enumerate(extractions)
    )

    result: StreamCompletionResult = chat(
        prompt_or_messages=[
            {
                "role": "user",
                "content": (
                    f"You are verifying a factual claim after detecting conflicting search snippets.\n\n"
                    f"Original Question: {query}\n\n"
                    f"Conflict Report: {json.dumps(conflict_report, indent=2)}\n\n"
                    f"Original Search Snippets:\n{search_snippets[:1000]}\n\n"
                    f"Verified Extractions from Primary Sources:\n{extraction_summary}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"1. Compare values across extractions\n"
                    f"2. If extractions agree, state the verified answer with sources\n"
                    f"3. If extractions disagree, present both values with source attribution\n"
                    f"4. NEVER invent values not present in extractions\n"
                    f"5. State confidence level based on source agreement"
                ),
            }
        ],
        model=Config.LLAMA_MODEL,
        temperature=0.0,
        max_tokens=768,
        project_name="qwen-studio-cross-reference",
        phoenix_url=Config.PHOENIX_URL,
    )
    return result.content.strip()
