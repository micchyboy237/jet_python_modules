# jet_python_modules/jet/agents/llama_cpp/live_rag_search/providers/llm.py

from abc import ABC, abstractmethod
from typing import Any, List

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.llm_utils import achat
from jet.logger import logger
from models import (
    ExtractedFacts,
    ExtractionResponse,
    LinkFilterResponse,
    SufficiencyResponse,
    SufficiencyResult,
    SufficiencyStatus,
)
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

PII_PATTERNS = ["ssn", "password", "api_key", "secret", "token"]


def _redact(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    lower = text.lower()
    for p in PII_PATTERNS:
        if p in lower:
            return "[REDACTED]"
    return text


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
        self, candidate_links: List[str], base_url: str, query: str, max_links: int
    ) -> List[str]: ...


class AnswerGenerator(ABC):
    @abstractmethod
    async def generate(
        self, query: str, memory_context: str, partial: bool = False
    ) -> str: ...


# --- IMPLEMENTATIONS USING STRUCTURED OUTPUT ---


class LlamaCppSufficiencyEvaluator(SufficiencyEvaluator):
    def __init__(self, model: str = LLM_MODEL, **kwargs: Any):
        self.model = model
        self.kwargs = kwargs

    async def evaluate(
        self, query: str, memory_context: str, new_context: str, source: str
    ) -> SufficiencyResult:
        with tracer.start_as_current_span(
            "live_rag.llm.sufficiency_evaluator",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                SpanAttributes.LLM_MODEL_NAME: self.model,
            },
        ) as span:
            prompt = (
                f"Goal: Determine if context answers: '{query}'\n\n"
                f"Memory:\n{memory_context[-1500:]}\n\n"
                f"New Context ({source}):\n{new_context[:2500]}\n\n"
                f"Decide if we have enough info."
            )

            try:
                result = await achat(
                    prompt_or_messages=prompt,
                    model=self.model,
                    response_format=SufficiencyResponse,
                    enable_thinking=False,
                    project_name="live-rag-eval",
                    **self.kwargs,
                )

                span.set_attribute(
                    SpanAttributes.OUTPUT_VALUE, _redact(result.content[:1000])
                )

                if result.structured and result.structured.success:
                    parsed: SufficiencyResponse = result.structured.parsed
                    status = (
                        SufficiencyStatus.COMPLETE
                        if parsed.status == "complete"
                        else SufficiencyStatus.INCOMPLETE
                    )
                    return SufficiencyResult(status=status, reasoning=parsed.reasoning)
                else:
                    logger.warning(
                        f"Structured output failed: {result.structured.error if result.structured else 'No result'}"
                    )
                    return SufficiencyResult(
                        status=SufficiencyStatus.ERROR, reasoning="Validation failed"
                    )

            except Exception as e:
                logger.error(f"Evaluator crash: {e}")
                return SufficiencyResult(
                    status=SufficiencyStatus.ERROR, reasoning=str(e)
                )


class LlamaCppFactExtractor(FactExtractor):
    def __init__(self, model: str = LLM_MODEL, **kwargs: Any):
        self.model = model
        self.kwargs = kwargs

    async def extract(
        self, content: str, query: str, existing_entity_ids: set[str]
    ) -> ExtractedFacts:
        with tracer.start_as_current_span(
            "live_rag.llm.fact_extractor",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                SpanAttributes.LLM_MODEL_NAME: self.model,
            },
        ) as span:
            prompt = (
                f"Task: Extract facts relevant to '{query}'.\n"
                f"Existing IDs (skip): {list(existing_entity_ids)[:20]}\n\n"
                f"Text:\n{content[:3000]}\n\n"
                f"Extract entities and their attributes."
            )

            try:
                result = await achat(
                    prompt_or_messages=prompt,
                    model=self.model,
                    response_format=ExtractionResponse,
                    enable_thinking=False,
                    project_name="live-rag-extract",
                    **self.kwargs,
                )

                span.set_attribute(
                    SpanAttributes.OUTPUT_VALUE, _redact(result.content[:2000])
                )

                if result.structured and result.structured.success:
                    parsed: ExtractionResponse = result.structured.parsed
                    return ExtractedFacts(entities=parsed.entities)
                else:
                    return ExtractedFacts()

            except Exception as e:
                logger.error(f"Extractor crash: {e}")
                return ExtractedFacts()


class LlamaCppInnerLinkFilter(InnerLinkFilter):
    def __init__(self, model: str = LLM_MODEL, **kwargs: Any):
        self.model = model
        self.kwargs = kwargs

    async def filter_links(
        self, candidate_links: List[str], base_url: str, query: str, max_links: int
    ) -> List[str]:
        if not candidate_links:
            return []

        with tracer.start_as_current_span(
            "live_rag.llm.link_filter",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                SpanAttributes.LLM_MODEL_NAME: self.model,
                "live_rag.candidate_links_count": len(candidate_links),
            },
        ) as span:
            links_str = "\n".join(candidate_links[:50])
            prompt = (
                f"Task: Select relevant URLs for '{query}'.\n"
                f"Candidates:\n{links_str}\n\n"
                f"Select up to {max_links}."
            )

            try:
                result = await achat(
                    prompt_or_messages=prompt,
                    model=self.model,
                    response_format=LinkFilterResponse,
                    enable_thinking=False,
                    project_name="live-rag-filter",
                    **self.kwargs,
                )

                span.set_attribute(SpanAttributes.OUTPUT_VALUE, result.content[:500])

                if result.structured and result.structured.success:
                    parsed: LinkFilterResponse = result.structured.parsed
                    valid = [u for u in parsed.urls if u in candidate_links]
                    return valid[:max_links]
                else:
                    return []

            except Exception as e:
                logger.error(f"Link filter crash: {e}")
                return []


class LlamaCppAnswerGenerator(AnswerGenerator):
    def __init__(self, model: str = LLM_MODEL, **kwargs: Any):
        self.model = model
        self.kwargs = kwargs

    async def generate(
        self, query: str, memory_context: str, partial: bool = False
    ) -> str:
        with tracer.start_as_current_span(
            "live_rag.llm.answer_generator",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                SpanAttributes.LLM_MODEL_NAME: self.model,
                SpanAttributes.INPUT_VALUE: _redact(query),
                "live_rag.partial_answer": partial,
            },
        ) as span:
            prompt = (
                f"Query: {query}\n"
                f"Facts: {memory_context}\n"
                f"{'Provide partial answer.' if partial else 'Provide complete answer.'}"
            )

            result = await achat(
                prompt_or_messages=prompt,
                model=self.model,
                project_name="live-rag-answer",
                **self.kwargs,
            )

            span.set_attribute(
                SpanAttributes.OUTPUT_VALUE, _redact(result.content[:3000])
            )
            return result.content
