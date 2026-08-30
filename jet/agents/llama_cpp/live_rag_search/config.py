from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyLimits:
    MAX_TOP_LEVEL_RESULTS: int = 10
    MAX_INNER_LINKS_PER_PAGE: int = 5
    MAX_TOTAL_SCRAPES: int = 30
    MAX_MEMORY_FACTS: int = 500
    SCRAPE_TIMEOUT_SEC: float = 10.0
    SUFFICIENCY_LLM_TIMEOUT_SEC: float = 15.0
    EXTRACTION_LLM_TIMEOUT_SEC: float = 15.0
    LINK_FILTER_LLM_TIMEOUT_SEC: float = 10.0


DEFAULT_LIMITS = SafetyLimits()
