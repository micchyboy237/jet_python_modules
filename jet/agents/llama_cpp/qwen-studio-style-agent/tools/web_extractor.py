import json
import logging

import trafilatura
from agent.config import Config
from jet.adapters.llama_cpp.llm_utils import chat

logger = logging.getLogger(__name__)

EXTRACTOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_extractor",
        "description": (
            "Extract and VERIFY specific information from a webpage. "
            "Use AFTER web_search to confirm facts before including them in responses. "
            "Returns structured JSON with error classification for adaptive retry logic."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL of the webpage"},
                "goal": {
                    "type": "string",
                    "description": (
                        "Precise extraction goal. Examples: "
                        "'Verify anime title, release year, and studio for [X]'; "
                        "'Confirm NBA season start date and official source'"
                    ),
                },
            },
            "required": ["url", "goal"],
        },
    },
}


def _classify_fetch_error(exc: Exception) -> str:
    """Classify network errors as permanent, temporary, or unknown."""
    exc_str = str(exc).lower()

    permanent_indicators = [
        "403",
        "404",
        "410",
        "captcha",
        "blocked",
        "forbidden",
        "not found",
        "access denied",
        "bot detection",
        "cloudflare",
    ]
    temporary_indicators = [
        "timeout",
        "connection reset",
        "503",
        "502",
        "504",
        "temporarily unavailable",
        "connection refused",
        "eof",
    ]

    if any(indicator in exc_str for indicator in permanent_indicators):
        return "permanent"
    if any(indicator in exc_str for indicator in temporary_indicators):
        return "temporary"
    return "unknown"


def _make_error_response(
    error_type: str,
    url: str,
    message: str,
    retry_recommended: bool,
    extra: dict | None = None,
) -> str:
    """Return standardized JSON error response for tool registry consumption."""
    payload = {
        "error": "extraction_failed",
        "error_type": error_type,
        "url": url,
        "message": message[:300],
        "retry_recommended": retry_recommended,
    }
    if extra:
        payload.update(extra)
    return json.dumps(payload)


def web_extractor(url: str, goal: str) -> str:
    """
    Extract goal-focused content from a URL with structured error handling.

    Returns either:
    - Plain text extracted content (success)
    - JSON string with error classification (failure)

    The ToolRegistry and search_manager parse this to decide retry vs fallback.
    """
    # === STAGE 1: FETCH WITH ERROR CLASSIFICATION ===
    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception as e:
        error_class = _classify_fetch_error(e)
        logger.warning(f"Fetch failed for {url}: {error_class} - {str(e)[:100]}")
        return _make_error_response(
            error_type=error_class,
            url=url,
            message=f"Network error: {type(e).__name__}: {str(e)}",
            retry_recommended=(error_class == "temporary"),
        )

    # Empty response = anti-bot block (treat as permanent)
    if not downloaded or len(downloaded.strip()) < 100:
        logger.warning(f"Empty/short response from {url}; likely anti-bot protection")
        return _make_error_response(
            error_type="permanent",
            url=url,
            message="Site returned empty or minimal content (likely anti-bot protection)",
            retry_recommended=False,
            extra={"content_length": len(downloaded) if downloaded else 0},
        )

    # === STAGE 2: CLEAN & EXTRACT TEXT ===
    try:
        cleaned = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            no_fallback=True,
            favor_precision=True,
        )
    except Exception as e:
        logger.error(f"Trafilatura extraction failed for {url}: {e}")
        return _make_error_response(
            error_type="unknown",
            url=url,
            message=f"Content parsing failed: {type(e).__name__}",
            retry_recommended=False,
        )

    if not cleaned or len(cleaned.strip()) < 50:
        logger.warning(f"No meaningful text extracted from {url}")
        return _make_error_response(
            error_type="permanent",
            url=url,
            message="No meaningful text content could be extracted from page",
            retry_recommended=False,
            extra={"raw_length": len(downloaded)},
        )

    # Truncate to preserve context budget
    cleaned = cleaned[: Config.EXTRACTOR_MAX_CHARS]

    # === STAGE 3: GOAL-FOCUSED LLM EXTRACTION ===
    try:
        result = chat(
            prompt_or_messages=[
                {
                    "role": "user",
                    "content": (
                        f"From this webpage content, extract ONLY information relevant to:\n"
                        f"GOAL: {goal}\n\nCONTENT:\n{cleaned}\n\n"
                        f"If the content does not contain information relevant to the goal, "
                        f'return exactly: {{"error": "not_found", "message": "<reason>"}}'
                    ),
                }
            ],
            model=Config.LLAMA_MODEL,
            temperature=0.0,
            max_tokens=1024,
            project_name="qwen-studio-extractor",
            phoenix_url=Config.PHOENIX_URL,
        )
        extracted_text = result.content.strip()
    except Exception as e:
        logger.error(f"LLM extraction failed for {url}: {e}")
        return _make_error_response(
            error_type="unknown",
            url=url,
            message=f"LLM summarization failed: {type(e).__name__}",
            retry_recommended=True,  # LLM failures are usually transient
        )

    # === STAGE 4: CHECK FOR LLM-REPORTED NOT FOUND ===
    # If the LLM itself says the content doesn't match the goal, treat as soft failure
    try:
        parsed_llm = json.loads(extracted_text)
        if isinstance(parsed_llm, dict) and parsed_llm.get("error") == "not_found":
            logger.info(f"LLM determined content irrelevant to goal for {url}")
            return _make_error_response(
                error_type="content_mismatch",
                url=url,
                message=parsed_llm.get(
                    "message", "Content does not address the extraction goal"
                ),
                retry_recommended=False,  # Don't retry same URL with same goal
            )
    except (json.JSONDecodeError, TypeError):
        pass  # Not JSON = normal text response, proceed

    # === SUCCESS: Return plain text for backward compatibility ===
    logger.info(f"Successfully extracted {len(extracted_text)} chars from {url}")
    return extracted_text
