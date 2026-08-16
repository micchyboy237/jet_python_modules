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
            "Managed search with automatic conflict detection, enforced verification, "
            "and adaptive fallback on extraction failures. Use INSTEAD of raw web_search "
            "for any factual query requiring accuracy."
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
    Enforced verification search pipeline with failure resilience:
    1. Dispatch: web_search for candidates
    2. Reflect: detect conflicts in snippets
    3. Verify: MANDATORY extraction with adaptive fallback on failures
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
        logger.info("✅ No conflicts detected; proceeding with snippet synthesis")
        return _synthesize_with_llm(query, search_results, conflict_report)

    # === STAGE 3: VERIFY (Mandatory Extraction with Fallback) ===
    logger.warning(
        "⚠️ Conflicts detected or verification forced; initiating mandatory extraction"
    )

    candidate_urls = _extract_top_urls(search_results, max_urls=3)
    if not candidate_urls:
        logger.warning(
            "Could not extract URLs for verification; falling back to snippets"
        )
        return _synthesize_with_llm(query, search_results, conflict_report)

    goal = f"Verify the exact answer to: {query}. Extract precise values with source context."
    extraction_result, tried_urls = _verify_with_fallback(candidate_urls, goal)

    # === STAGE 4: AGGREGATE & CROSS-REFERENCE ===
    return _cross_reference_and_synthesize(
        query=query,
        search_snippets=search_results,
        extraction_result=extraction_result,
        tried_urls=tried_urls,
        conflict_report=conflict_report,
    )


def _extract_top_urls(search_results: str, max_urls: int = 3) -> list[str]:
    """Parse URLs from formatted search results, filtering low-relevance noise."""
    urls = []
    for line in search_results.split("\n"):
        if line.strip().startswith("URL:"):
            url = line.replace("URL:", "").strip()
            # Skip dictionary/definition sites that pollute results
            if url.startswith("http") and not any(
                domain in url
                for domain in [
                    "dictionary.cambridge.org",
                    "merriam-webster.com",
                    "wikipedia.org/wiki/Dune_(2021",
                ]
            ):
                urls.append(url)
                if len(urls) >= max_urls:
                    break
    return urls


def _verify_with_fallback(
    candidate_urls: list[str], goal: str
) -> tuple[str, list[str]]:
    """
    Try extraction on URLs in priority order until one succeeds or all exhausted.
    Respects error classification from web_extractor to avoid retrying permanent failures.
    """
    tried: list[str] = []

    for i, url in enumerate(candidate_urls):
        tried.append(url)
        logger.info(
            f"📄 Extraction attempt {i + 1}/{len(candidate_urls)}: {url[:70]}..."
        )

        result = web_extractor(url=url, goal=goal)

        # Parse structured error response if present
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict) and "error" in parsed:
                error_type = parsed.get("error_type", "unknown")
                retry_ok = parsed.get("retry_recommended", False)

                if not retry_ok or error_type in (
                    "permanent",
                    "content_mismatch",
                    "url_blacklisted",
                ):
                    logger.warning(
                        f"Permanent/non-retryable failure on {url}: {error_type}. "
                        f"Trying next candidate..."
                    )
                    continue

                # Temporary failure but no more candidates
                if i == len(candidate_urls) - 1:
                    logger.warning(
                        f"All candidates exhausted. Last error: {error_type}"
                    )
                    return result, tried

                # Temporary failure with more candidates available
                logger.info(f"Temporary failure on {url}; trying next candidate")
                continue

        except (json.JSONDecodeError, TypeError):
            # Non-JSON result = successful plain-text extraction
            logger.info(f"✅ Successful extraction from {url}")
            return result, tried

    # All URLs exhausted without success
    logger.error(f"All {len(tried)} candidate URLs failed verification")
    return json.dumps(
        {
            "error": "all_candidates_exhausted",
            "tried_urls": tried,
            "message": "Could not verify information from primary sources after trying all candidates.",
            "recommendation": "Answer based on search snippets with explicit uncertainty disclaimer.",
        }
    ), tried


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
    extraction_result: str,
    tried_urls: list[str],
    conflict_report: dict,
) -> str:
    """Cross-reference extracted content and produce verified answer with uncertainty handling."""

    # Check if extraction fully failed
    try:
        parsed_extraction = json.loads(extraction_result)
        if isinstance(parsed_extraction, dict) and "error" in parsed_extraction:
            # Full extraction failure — force uncertainty-aware synthesis from snippets only
            logger.warning(
                "Extraction phase failed completely; synthesizing with explicit uncertainty"
            )
            result: StreamCompletionResult = chat(
                prompt_or_messages=[
                    {
                        "role": "user",
                        "content": (
                            f"You attempted to verify a factual claim but ALL primary source extractions failed.\n\n"
                            f"Original Question: {query}\n\n"
                            f"URLs Tried (all failed): {tried_urls}\n\n"
                            f"Failure Reason: {parsed_extraction.get('message', 'Unknown')}\n\n"
                            f"Available Search Snippets (UNVERIFIED):\n{search_snippets[:1500]}\n\n"
                            f"INSTRUCTIONS:\n"
                            f"1. State clearly that primary source verification failed\n"
                            f"2. Provide best available answer from snippets WITH explicit uncertainty markers\n"
                            f"3. List which sources could not be accessed\n"
                            f"4. NEVER present snippet data as verified fact\n"
                            f"5. Recommend user verify directly at authoritative sources"
                        ),
                    }
                ],
                model=Config.LLAMA_MODEL,
                temperature=0.0,
                max_tokens=768,
                project_name="qwen-studio-fallback-synthesis",
                phoenix_url=Config.PHOENIX_URL,
            )
            return result.content.strip()
    except (json.JSONDecodeError, TypeError):
        pass  # Normal text extraction succeeded

    # Standard cross-reference path (extraction succeeded)
    result: StreamCompletionResult = chat(
        prompt_or_messages=[
            {
                "role": "user",
                "content": (
                    f"You are verifying a factual claim after detecting conflicting search snippets.\n\n"
                    f"Original Question: {query}\n\n"
                    f"Conflict Report: {json.dumps(conflict_report, indent=2)}\n\n"
                    f"Original Search Snippets:\n{search_snippets[:1000]}\n\n"
                    f"Verified Extraction from Primary Source(s):\n{extraction_result}\n\n"
                    f"URLs Successfully Extracted: {tried_urls}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"1. Compare values across extractions and snippets\n"
                    f"2. If extractions agree, state the verified answer with sources\n"
                    f"3. If extractions disagree with snippets, prefer extraction but note discrepancy\n"
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
