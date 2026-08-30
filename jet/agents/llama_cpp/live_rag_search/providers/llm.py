import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from jet.adapters.langchain.chat_llama_cpp import ChatLlamaCpp
from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.logger import logger
from langchain_core.messages import HumanMessage, SystemMessage
from models import ExtractedFacts, SufficiencyResult, SufficiencyStatus
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


def _extract_usage(response_obj) -> Dict[str, int]:
    """Extract token usage from LangChain AIMessageChunk or similar."""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        um = getattr(response_obj, "usage_metadata", None)
        if um:
            usage["prompt_tokens"] = um.get("input_tokens", 0)
            usage["completion_tokens"] = um.get("output_tokens", 0)
            usage["total_tokens"] = um.get("total_tokens", 0)
    except Exception:
        pass
    return usage


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


class LlamaCppSufficiencyEvaluator(SufficiencyEvaluator):
    def __init__(self, model: str = LLM_MODEL, **kwargs: Any):
        self.llm = ChatLlamaCpp(model=model, enable_thinking=False, **kwargs)

    async def evaluate(
        self, query: str, memory_context: str, new_context: str, source: str
    ) -> SufficiencyResult:
        with tracer.start_as_current_span(
            "live_rag.llm.sufficiency_evaluator",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                SpanAttributes.LLM_MODEL_NAME: self.llm._model,
                SpanAttributes.INPUT_VALUE: _redact(
                    f"Query: {query}\nSource: {source}"
                ),
            },
        ) as span:
            # Truncate contexts to fit local model window
            mem_snippet = (
                memory_context[-2000:] if len(memory_context) > 2000 else memory_context
            )
            new_snippet = new_context[:3000]

            prompt = (
                f"Goal: Determine if we have enough info to answer: '{query}'\n\n"
                f"Accumulated Memory:\n{mem_snippet}\n\n"
                f"New Context ({source}):\n{new_snippet}\n\n"
                f'Output JSON only: {{"status": "complete"|"incomplete", "reasoning": "..."}}\n'
                f"If status is incomplete, explain what specific facts are missing."
            )

            messages = [HumanMessage(content=prompt)]
            collected = []
            final_chunk = None

            # Stream output
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                    collected.append(chunk.content)
                final_chunk = chunk
            print(flush=True)

            raw = "".join(collected).strip()
            usage = _extract_usage(final_chunk) if final_chunk else {}

            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT, usage.get("prompt_tokens", 0)
            )
            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                usage.get("completion_tokens", 0),
            )
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, _redact(raw[:1000]))

            try:
                start = raw.find("{")
                end = raw.rfind("}") + 1
                data = json.loads(raw[start:end]) if start != -1 and end > 0 else {}
                status = (
                    SufficiencyStatus.COMPLETE
                    if data.get("status") == "complete"
                    else SufficiencyStatus.INCOMPLETE
                )
                return SufficiencyResult(
                    status=status, reasoning=data.get("reasoning", "")
                )
            except Exception as e:
                logger.warning(f"Sufficiency parse error: {e}")
                return SufficiencyResult(
                    status=SufficiencyStatus.ERROR, reasoning=str(e)
                )


class LlamaCppFactExtractor(FactExtractor):
    def __init__(self, model: str = LLM_MODEL, **kwargs: Any):
        self.llm = ChatLlamaCpp(model=model, enable_thinking=False, **kwargs)

    async def extract(
        self, content: str, query: str, existing_entity_ids: set[str]
    ) -> ExtractedFacts:
        with tracer.start_as_current_span(
            "live_rag.llm.fact_extractor",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                SpanAttributes.LLM_MODEL_NAME: self.llm._model,
            },
        ) as span:
            prompt = (
                f"Extract key entities/facts relevant to '{query}' from the text below.\n"
                f"Existing IDs (skip these): {list(existing_entity_ids)[:50]}\n"
                f"Text: {content[:4000]}\n"
                f'Return JSON: {{"entities": {{"entity_id": {{"fact_key": "value"}}}}}}'
            )

            messages = [HumanMessage(content=prompt)]
            collected = []
            final_chunk = None

            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                    collected.append(chunk.content)
                final_chunk = chunk
            print(flush=True)

            raw = "".join(collected).strip()
            usage = _extract_usage(final_chunk) if final_chunk else {}

            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT, usage.get("prompt_tokens", 0)
            )
            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                usage.get("completion_tokens", 0),
            )
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, _redact(raw[:2000]))

            try:
                start = raw.find("{")
                end = raw.rfind("}") + 1
                data = json.loads(raw[start:end]) if start != -1 and end > 0 else {}
                return ExtractedFacts(entities=data.get("entities", {}))
            except Exception as e:
                logger.warning(f"Fact extraction parse error: {e}")
                return ExtractedFacts()


class LlamaCppInnerLinkFilter(InnerLinkFilter):
    def __init__(self, model: str = LLM_MODEL, **kwargs: Any):
        self.llm = ChatLlamaCpp(model=model, enable_thinking=False, **kwargs)

    async def filter_links(
        self, candidate_links: List[str], base_url: str, query: str, max_links: int
    ) -> List[str]:
        if not candidate_links:
            return []

        with tracer.start_as_current_span(
            "live_rag.llm.link_filter",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                SpanAttributes.LLM_MODEL_NAME: self.llm._model,
                "live_rag.candidate_links_count": len(candidate_links),
            },
        ) as span:
            # Limit candidates to prevent prompt overflow
            links_str = "\n".join(candidate_links[:50])

            prompt = (
                f"Task: Select the most relevant URLs to answer the query.\n"
                f"Query: '{query}'\n"
                f"Candidate URLs:\n{links_str}\n\n"
                f"Instructions:\n"
                f"1. Return ONLY a JSON list of URLs.\n"
                f"2. Select at most {max_links} URLs.\n"
                f"3. Prioritize pages likely containing detailed info (e.g., specific anime pages, lists with dates).\n"
                f"4. Ignore navigation, login, or generic home pages.\n"
                f'Output format: ["url1", "url2"]'
            )

            messages = [HumanMessage(content=prompt)]
            collected = []
            final_chunk = None

            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                    collected.append(chunk.content)
                final_chunk = chunk
            print(flush=True)

            raw = "".join(collected).strip()
            usage = _extract_usage(final_chunk) if final_chunk else {}

            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT, usage.get("prompt_tokens", 0)
            )
            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                usage.get("completion_tokens", 0),
            )
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, raw[:500])

            try:
                start = raw.find("[")
                end = raw.rfind("]") + 1
                if start != -1 and end > 0:
                    links = json.loads(raw[start:end])
                    # Validate against candidates to prevent hallucination
                    valid_links = [l for l in links if l in candidate_links]
                    return valid_links[:max_links]
            except Exception as e:
                logger.warning(f"Link filter parse error: {e}")

            return []


class LlamaCppAnswerGenerator(AnswerGenerator):
    def __init__(self, model: str = LLM_MODEL, **kwargs: Any):
        self.llm = ChatLlamaCpp(model=model, enable_thinking=False, **kwargs)

    async def generate(
        self, query: str, memory_context: str, partial: bool = False
    ) -> str:
        with tracer.start_as_current_span(
            "live_rag.llm.answer_generator",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                SpanAttributes.LLM_MODEL_NAME: self.llm._model,
                SpanAttributes.INPUT_VALUE: _redact(query),
                "live_rag.partial_answer": partial,
            },
        ) as span:
            system_msg = (
                "You are a precise RAG assistant. Answer based ONLY on accumulated facts. "
                "If partial=True, provide what you have so far and note gaps."
            )
            user_msg = (
                f"Query: {query}\n"
                f"Accumulated Facts: {memory_context}\n"
                f"{'Provide a partial answer noting missing info.' if partial else 'Provide a complete answer.'}"
            )

            messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=user_msg),
            ]
            collected = []
            final_chunk = None

            print("\n🤖 Final Answer: ", end="", flush=True)
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                    collected.append(chunk.content)
                final_chunk = chunk
            print(flush=True)

            answer = "".join(collected)
            usage = _extract_usage(final_chunk) if final_chunk else {}

            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT, usage.get("prompt_tokens", 0)
            )
            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                usage.get("completion_tokens", 0),
            )
            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL, usage.get("total_tokens", 0)
            )
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, _redact(answer[:3000]))

            return answer
