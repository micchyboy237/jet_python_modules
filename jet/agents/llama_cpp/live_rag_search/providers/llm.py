from abc import ABC, abstractmethod
from typing import List

from models import ExtractedFacts, SufficiencyResult


class SufficiencyEvaluator(ABC):
    @abstractmethod
    async def evaluate(
        self, query: str, memory_context: str, new_context: str, source: str
    ) -> SufficiencyResult: ...


class FactExtractor(ABC):
    @abstractmethod
    async def extract(
        self, content: str, query: str, existing_entity_ids: set[str]
    ) -> ExtractedFacts: ...


class InnerLinkFilter(ABC):
    @abstractmethod
    async def filter_links(
        self, page_content: str, base_url: str, query: str, max_links: int
    ) -> List[str]: ...


class AnswerGenerator(ABC):
    @abstractmethod
    async def generate(
        self, query: str, memory_context: str, partial: bool = False
    ) -> str: ...


# === OpenAI Implementations (placeholders) ===


class OpenAISufficiencyEvaluator(SufficiencyEvaluator):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model

    async def evaluate(
        self, query, memory_context, new_context, source
    ) -> SufficiencyResult:
        # TODO: Structured output call with Pydantic schema
        # Prompt must include explicit completeness criteria derived from query
        raise NotImplementedError


class OpenAIFactExtractor(FactExtractor):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model

    async def extract(self, content, query, existing_entity_ids) -> ExtractedFacts:
        # TODO: Extract structured facts, skip existing entity IDs
        raise NotImplementedError


class OpenAIInnerLinkFilter(InnerLinkFilter):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model

    async def filter_links(self, page_content, base_url, query, max_links) -> List[str]:
        # TODO: Lightweight classification of relevant inner links
        raise NotImplementedError


class OpenAIAnswerGenerator(AnswerGenerator):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model

    async def generate(self, query, memory_context, partial=False) -> str:
        # TODO: Final answer generation with transparency on gaps if partial=True
        raise NotImplementedError
