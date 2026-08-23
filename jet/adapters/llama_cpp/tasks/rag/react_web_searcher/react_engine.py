"""ReAct loop engine using llm_utils.achat with full feature utilization.
✅ IMPROVEMENTS:
- Validation feedback loop: re-searches on critical failure instead of returning bad answer
- Source population: extracts source URLs from AgentStep metadata for citation
- Re-search cap: prevents infinite loops from persistent validation failures
- Dynamic max_tokens: derives generation budget from model context window
- ✅ NEW: AccumulationMemory integration for token-accurate context tracking
- ✅ NEW: Intermediate sufficiency evaluation gate inside ReAct loop
- ✅ NEW: Programmatic list-intent guardrail via tool registry
- ✅ NEW: List-intent rules injected into user message for small model adherence
- ✅ NEW: Forced synthesis when memory budget exhausted or target reached
- ✅ NEW: Token-aware context truncation in sufficiency checks (replaces char-ratio)
- ✅ CANONICAL FIX: Explicit Thought: parsing from agent output for auditable reasoning
- ✅ CANONICAL FIX: Action-based sufficiency inference replaces static keyword heuristic
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from jet.adapters.llama_cpp.config import EMBED_MODEL_LG, LLM_MODEL, RERANK_MODEL
from jet.adapters.llama_cpp.factory import get_async_llm_client
from jet.adapters.llama_cpp.llm_utils import achat
from jet.adapters.llama_cpp.model_utils import get_model_ctx_embd_size
from jet.adapters.llama_cpp.token_utils import count_tokens
from openai import AsyncOpenAI

from .memory import AccumulationMemory
from .query_analyzer import QueryAnalyzer
from .tools import get_tool_definitions, get_tool_registry, truncate_to_tokens
from .types import AgentStep, FinalAnswer, QueryComplexity, QueryIntent, SearchResult
from .validator import PostAnswerValidator

logger = logging.getLogger(__name__)

# Regex for extracting explicit Thought: traces from agent content
_THOUGHT_PATTERN = re.compile(
    r"Thought:\s*(.*?)(?=\nAction:|\nObservation:|$)", re.DOTALL | re.IGNORECASE
)


def _extract_thought_from_content(content: str) -> str:
    """Extract explicit Thought: trace from LLM content.
    Returns empty string if no Thought: prefix found (native tool-call mode).
    This enables backward compatibility: when the model uses native function-calling
    without explicit Thought: prefixes, behavior is identical to before.
    """
    if not content:
        return ""
    match = _THOUGHT_PATTERN.search(content)
    return match.group(1).strip() if match else ""


SYSTEM_PROMPT = """\
You are a thorough web research agent. Your goal is to answer the user's question \
accurately and completely using web search.
You have access to three tools:
1. searxng_search - Search the web for information
2. read_url - Read the full content of a specific web page (supports focused extraction via 'query' param)
3. synthesize - Combine your findings into a final answer

## GENERAL RULES
1. ⚠️ CRITICAL: Search snippets are summaries and often lack detail.
   You MUST use read_url on at least 1-2 promising results to verify facts
   and gather comprehensive details before synthesizing.
2. When calling read_url, ALWAYS pass the original query or sub-query
   to focus extraction on relevant sections.
3. When you have enough VERIFIED information, call synthesize to produce the final answer.
4. Always cite your sources.
5. Do NOT make up information. Only use what you find via search AND verify via read_url.
6. If you cannot find sufficient information after multiple searches and page reads, say so.
7. Call synthesize ONLY when ready to produce the final answer.
8. After each search or page read, carefully evaluate whether you have enough
   information to answer the query COMPLETELY before deciding next steps.

## REASONING FORMAT
Before EVERY tool call, you MUST write a Thought: line explaining:
- What you have learned so far
- Whether you have ENOUGH INFORMATION to answer completely
- What specific gaps remain (if any)
- Why you chose this particular next action

Example:
Thought: I found that pandas eat bamboo, but I still need their habitat range.
       I have enough on diet but missing geography. Searching for habitat info.
Action: searxng_search
"""

_SUFFICIENCY_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "ContextSufficiencyCheck",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "is_sufficient": {
                    "type": "boolean",
                    "description": "True if accumulated context fully answers the query",
                },
                "missing_info": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific pieces of information still needed (empty if sufficient)",
                },
                "next_action": {
                    "type": "string",
                    "enum": [
                        "synthesize",
                        "search_new_query",
                        "read_next_link",
                        "stop",
                    ],
                    "description": (
                        "Recommended next step. 'synthesize' if sufficient. "
                        "'search_new_query' if gaps require a different query. "
                        "'read_next_link' if existing search results have unread promising links. "
                        "'stop' if further searching is unlikely to help."
                    ),
                },
                "suggested_query": {
                    "type": "string",
                    "description": "New search query to fill gaps (only when next_action=search_new_query)",
                },
            },
            "required": ["is_sufficient", "missing_info", "next_action"],
        },
    },
}

_MAX_VALIDATION_RETRIES = 2
_DEFAULT_REACT_MAX_TOKENS = 4096
_MAX_SUFFICIENCY_CHECKS = 8


def _get_react_max_tokens(model: str) -> int:
    """Derive max_tokens for ReAct loop from model context window.
    Reserves 25% of context for prompt/tools/observations, uses remaining
    75% as max generation tokens. Falls back to 4096 on failure.
    """
    try:
        ctx_info = get_model_ctx_embd_size(model)
        ctx = ctx_info.get("ctx", 0)
        if ctx > 0:
            max_gen = int(ctx * 0.75)
            logger.debug(
                "📏 ReAct max_tokens derived: ctx=%d → max_gen=%d",
                ctx,
                max_gen,
            )
            return max(max_gen, 1024)
    except Exception as exc:
        logger.warning(
            "⚠️ Could not derive max_tokens for %s (%s: %s); using default %d",
            model,
            type(exc).__name__,
            exc,
            _DEFAULT_REACT_MAX_TOKENS,
        )
    return _DEFAULT_REACT_MAX_TOKENS


def _build_list_intent_user_message(query: str, refined_query: str) -> str:
    """Build user message with list-intent rules embedded directly.
    Small models follow user-message instructions more reliably than
    system-prompt negative constraints. This moves the critical list
    guardrail from system prompt to user message.
    """
    return (
        f"Question: {refined_query}\n"
        f"⚠️ IMPORTANT LIST-QUERY RULES:\n"
        f"- This is a LIST/RANKING query. Search for CURATED LISTS only.\n"
        f"- ✅ DO: Search for 'best [topic] list [year]' or 'top [topic] ranking [year]'\n"
        f"- ✅ DO: Use read_url on the best list page to extract the full ranking\n"
        f"- ❌ NEVER: Search for individual items/titles one-by-one\n"
        f"- ❌ NEVER: Decompose into per-entity sub-queries\n"
        f"- If the first list is incomplete, search for ANOTHER LIST, not individual items\n"
        f"- When you have enough items from list pages, call synthesize immediately\n"
        f"Search for information and provide a complete, accurate answer."
    )


class ReactEngine:
    """Orchestrates the full ReAct search pipeline.

    Utilizes all applicable llm_utils.achat features:
    - session_id: All calls within one search share a Phoenix conversation thread
    - seed: Analyzer uses fixed seed for reproducible decomposition
    - stop: Prevents runaway generation beyond current turn
    - client: Single shared AsyncOpenAI client avoids per-call overhead
    - finish_reason: Detects truncation on final answer
    - step_tracker: Mutable list passed to tool wrappers for accurate step counting

    ✅ NEW: AccumulationMemory integration with token-accurate budgeting.
    ✅ NEW: Intermediate context sufficiency evaluation gate.
    ✅ NEW: Programmatic list-intent guardrail via tool registry.
    ✅ NEW: List-intent rules in user message for small model adherence.
    ✅ NEW: Forced synthesis when memory budget exhausted or target reached.
    ✅ NEW: Token-aware context truncation in sufficiency checks.
    ✅ CANONICAL FIX: Explicit Thought: parsing for auditable reasoning traces.
    ✅ CANONICAL FIX: Action-based sufficiency inference (replaces static keywords).
    ✅ EXISTING: Validation feedback loop re-searches on critical failure.
    ✅ EXISTING: Source population from AgentStep metadata.
    ✅ EXISTING: Dynamic max_tokens derived from model context window.
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        max_iterations: int = 10,
        enable_validation: bool = True,
        embed_model: str = EMBED_MODEL_LG,
        rerank_model: str = RERANK_MODEL,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.enable_validation = enable_validation
        self.embed_model = embed_model
        self.rerank_model = rerank_model
        self.analyzer = QueryAnalyzer(model=model)
        self.validator = PostAnswerValidator(model=model) if enable_validation else None
        self.tool_definitions = get_tool_definitions()
        self._client: AsyncOpenAI = get_async_llm_client()

    @staticmethod
    def _infer_sufficiency_from_action(
        result: Any,
        answer_text: str,
    ) -> dict[str, Any] | None:
        """Infer sufficiency from the agent's actual tool-call decision.

        Reads the tool calls emitted by the model in this turn to determine
        what the agent decided to do next. This is always consistent with
        the agent's behavior and requires zero extra tokens or parsing.

        Returns same schema as _evaluate_context_sufficiency for drop-in use.
        Returns None when the agent's intent is ambiguous (no tool calls and
        no content), signaling the caller to fall back to structured LLM check.
        """
        # Agent explicitly chose to synthesize → sufficient
        if result.has_tool_calls:
            tool_names = [tc.name for tc in result.tool_calls]
            if "synthesize" in tool_names:
                return {
                    "is_sufficient": True,
                    "missing_info": [],
                    "next_action": "synthesize",
                    "suggested_query": "",
                }
            if "read_url" in tool_names:
                return {
                    "is_sufficient": False,
                    "missing_info": ["Agent chose to read another URL"],
                    "next_action": "read_next_link",
                    "suggested_query": "",
                }
            if "searxng_search" in tool_names:
                # Extract the query the agent chose as the suggested next query
                search_tc = next(
                    (tc for tc in result.tool_calls if tc.name == "searxng_search"),
                    None,
                )
                suggested = ""
                if search_tc and isinstance(search_tc.arguments, dict):
                    suggested = search_tc.arguments.get("query", "")
                return {
                    "is_sufficient": False,
                    "missing_info": ["Agent chose to search again"],
                    "next_action": "search_new_query",
                    "suggested_query": suggested,
                }
            # Unknown tool → ambiguous
            return None

        # Agent returned content with no tool call → implicit synthesis attempt
        if answer_text and answer_text.strip():
            return {
                "is_sufficient": True,
                "missing_info": [],
                "next_action": "synthesize",
                "suggested_query": "",
            }

        # No tool calls and no content → ambiguous, fall back to structured check
        return None

    async def _evaluate_context_sufficiency(
        self,
        query: str,
        memory: AccumulationMemory,
        session_id: str,
    ) -> dict[str, Any]:
        """Structured sufficiency check using memory's accumulated contexts.

        ✅ CHANGED: Uses truncate_to_tokens for accurate budget fitting
        instead of char-ratio estimation.

        Returns dict with keys: is_sufficient, missing_info, next_action, suggested_query.
        On failure, defaults to continuing the loop (conservative fallback).
        """
        contexts = memory.get_accumulated_contexts()
        if not contexts:
            return {
                "is_sufficient": False,
                "missing_info": ["No context collected yet"],
                "next_action": "search_new_query",
                "suggested_query": "",
            }

        combined = "\n---\n".join(contexts)
        judge_budget = memory.remaining_token_budget + 2048
        combined_tokens = count_tokens(combined, model=self.model)

        if combined_tokens > judge_budget:
            original_tokens = combined_tokens
            combined = truncate_to_tokens(
                combined, judge_budget, model=self.model, suffix="\n...[truncated]"
            )
            new_tokens = count_tokens(combined, model=self.model)
            logger.debug(
                "✂️ Truncated sufficiency context: %d → %d tokens (budget=%d)",
                original_tokens,
                new_tokens,
                judge_budget,
            )

        messages = [
            {
                "role": "user",
                "content": (
                    f"Evaluate whether the accumulated context below is SUFFICIENT "
                    f"to fully answer this query.\n"
                    f"Query: {query}\n"
                    f"Query Intent: {memory.intent.value}\n"
                    f"List Items Collected: {memory.list_item_count}\n"
                    f"Accumulated Context ({memory.accumulated_tokens} tokens):\n{combined}\n"
                    f"Respond with valid JSON matching the ContextSufficiencyCheck schema."
                ),
            },
        ]

        try:
            result = await achat(
                prompt_or_messages=messages,
                model=self.model,
                project_name="react-sufficiency-check",
                temperature=0.0,
                max_tokens=512,
                response_format=_SUFFICIENCY_SCHEMA,
                enable_thinking=False,
                capture_content=True,
                session_id=session_id,
                client=self._client,
            )

            if result.structured and result.structured.success:
                parsed = result.structured.parsed
                if isinstance(parsed, dict):
                    logger.info(
                        "📊 Sufficiency: sufficient=%s next=%s gaps=%d tokens=%d/%d",
                        parsed.get("is_sufficient"),
                        parsed.get("next_action"),
                        len(parsed.get("missing_info", [])),
                        memory.accumulated_tokens,
                        memory.max_tokens,
                    )
                    return parsed

            logger.warning(
                "⚠️ Sufficiency parse failed: %s",
                result.structured.error
                if result.structured
                else "No structured output",
            )
        except Exception as exc:
            logger.warning("⚠️ Sufficiency exception: %s", exc)

        return {
            "is_sufficient": False,
            "missing_info": [],
            "next_action": "search_new_query",
            "suggested_query": "",
        }

    async def search(self, query: str) -> FinalAnswer:
        """Run the full ReAct web search pipeline for a query."""
        logger.info("🚀 Starting ReAct search for: %r", query[:80])
        session_id = f"react-{uuid.uuid4().hex[:12]}"
        logger.debug("🧵 Session ID: %s", session_id)

        logger.debug("📋 Step 1: Analyzing query complexity")
        analysis = await self.analyzer.analyze(
            query, session_id=session_id, client=self._client
        )
        logger.info(
            "📋 Analysis result: complexity=%s, intent=%s, sub_queries=%d, refined=%r",
            analysis.complexity.value,
            analysis.intent.value,
            len(analysis.sub_queries),
            analysis.refined_query[:60],
        )

        target_list_size = 10 if analysis.intent == QueryIntent.LIST else None
        memory = AccumulationMemory(
            model=self.model,
            intent=analysis.intent,
            target_list_size=target_list_size,
        )
        logger.info(
            "🧠 Memory initialized: max_tokens=%d, intent=%s, target_list=%s",
            memory.max_tokens,
            memory.intent.value,
            target_list_size,
        )

        if analysis.complexity == QueryComplexity.COMPLEX and analysis.sub_queries:
            user_content = (
                f"Original Question: {query}\n"
                f"This is a complex question. Break it down and search for each sub-query:\n"
                + "\n".join(f"- {sq}" for sq in analysis.sub_queries)
                + f"\nRefined query: {analysis.refined_query}\n"
                f"Search each sub-query, gather findings, then synthesize a complete answer."
            )
            logger.debug(
                "📝 Complex query: %d sub-queries injected into prompt",
                len(analysis.sub_queries),
            )
        elif analysis.intent == QueryIntent.LIST:
            user_content = _build_list_intent_user_message(
                query, analysis.refined_query
            )
            logger.debug("📝 List query: intent rules injected into user message")
        else:
            user_content = (
                f"Question: {analysis.refined_query}\n"
                f"Search for information and provide a complete, accurate answer."
            )
            logger.debug("📝 Simple query: using refined query directly")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        steps: list[AgentStep] = []
        bound_registry = get_tool_registry(
            step_tracker=steps,
            embed_model=self.embed_model,
            rerank_model=self.rerank_model,
            query_intent=analysis.intent,
            memory=memory,
            last_assistant_content="",
        )

        react_max_tokens = _get_react_max_tokens(self.model)
        logger.info(
            "🔄 Starting ReAct loop (max_iterations=%d, tools=%d, session=%s, max_tokens=%d)",
            self.max_iterations,
            len(bound_registry),
            session_id,
            react_max_tokens,
        )

        validation_retries = 0
        sufficiency_checks = 0
        answer_text = ""
        truncated = False
        total_tokens = 0
        eval_result = None
        confidence = "high"

        while True:
            result = await achat(
                prompt_or_messages=messages,
                model=self.model,
                project_name="react-web-searcher",
                temperature=0.3,
                max_tokens=react_max_tokens,
                tools=self.tool_definitions,
                tool_choice="auto",
                tool_registry=bound_registry,
                max_tool_rounds=self.max_iterations,
                enable_thinking=False,
                capture_content=True,
                session_id=session_id,
                client=self._client,
            )

            truncated = result.finish_reason == "length"
            total_tokens = result.usage.get("total_tokens", 0) if result.usage else 0
            answer_text = result.content or ""

            logger.info(
                "✅ ReAct loop iteration: %d steps, %d tokens, %d chars, truncated=%s, finish=%s",
                len(steps),
                total_tokens,
                len(answer_text),
                truncated,
                result.finish_reason,
            )

            for i, step in enumerate(steps, 1):
                logger.debug(
                    "   Step %d/%d: %s(%s) → %d chars observation",
                    i,
                    len(steps),
                    step.action,
                    list(step.action_input.keys()),
                    len(step.observation),
                )

            if truncated:
                logger.warning(
                    "⚠️ ReAct loop truncated at max_tokens=%d. "
                    "Final answer may be incomplete.",
                    react_max_tokens,
                )

            # ✅ CANONICAL FIX: Rebuild registry with extracted thought for next turn
            bound_registry = get_tool_registry(
                step_tracker=steps,
                embed_model=self.embed_model,
                rerank_model=self.rerank_model,
                query_intent=analysis.intent,
                memory=memory,
                last_assistant_content=answer_text,
            )

            if memory.should_force_synthesis() and not answer_text:
                logger.info(
                    "🛑 Memory requests forced synthesis (tokens=%d/%d, items=%d)",
                    memory.accumulated_tokens,
                    memory.max_tokens,
                    memory.list_item_count,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "BUDGET EXHAUSTED OR TARGET REACHED. "
                            "Call synthesize NOW with all accumulated findings. "
                            "If information is insufficient, say so explicitly."
                        ),
                    }
                )
                continue

            # ✅ ACTION-BASED SUFFICIENCY: Infer from agent's actual tool-call decision.
            # Falls back to structured LLM check only when action is ambiguous.
            if (
                memory.num_contexts > 0
                and not answer_text
                and sufficiency_checks < _MAX_SUFFICIENCY_CHECKS
                and memory.get_sufficiency_snapshot() is None
            ):
                sufficiency_checks += 1

                sufficiency = self._infer_sufficiency_from_action(result, answer_text)

                if sufficiency is None:
                    logger.debug(
                        "🔍 Action-based sufficiency ambiguous, running structured check #%d "
                        "(%d contexts, %d tokens)",
                        sufficiency_checks,
                        memory.num_contexts,
                        memory.accumulated_tokens,
                    )
                    sufficiency = await self._evaluate_context_sufficiency(
                        query=query,
                        memory=memory,
                        session_id=session_id,
                    )
                else:
                    logger.info(
                        "🎯 Action-based sufficiency: sufficient=%s next=%s (saved 1 LLM call)",
                        sufficiency["is_sufficient"],
                        sufficiency["next_action"],
                    )

                memory.update_sufficiency(**sufficiency)

                next_action = sufficiency.get("next_action", "search_new_query")
                is_sufficient = sufficiency.get("is_sufficient", False)

                if is_sufficient or next_action == "synthesize":
                    logger.info(
                        "✅ Sufficiency PASSED — injecting synthesize directive"
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "SUFFICIENCY CHECK PASSED. You have enough context to answer. "
                                "Call synthesize NOW with all accumulated findings."
                            ),
                        }
                    )
                elif next_action == "stop":
                    logger.info(
                        "🛑 Sufficiency recommends STOP — injecting synthesize directive"
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Further searching is unlikely to yield better results. "
                                "Call synthesize NOW with whatever findings you have. "
                                "If information is insufficient, say so explicitly."
                            ),
                        }
                    )
                elif next_action == "search_new_query":
                    suggested = sufficiency.get("suggested_query", "")
                    missing = sufficiency.get("missing_info", [])
                    gap_desc = "; ".join(missing[:3]) if missing else "unspecified gaps"
                    logger.info(
                        "🔎 Sufficiency gap: %s — suggesting: %r",
                        gap_desc,
                        suggested[:80] if suggested else "(none)",
                    )
                    guidance = f"GAP DETECTED: {gap_desc}."
                    if suggested:
                        guidance += f" Search for: {suggested}"
                    else:
                        guidance += " Generate a new search query to fill these gaps."
                    messages.append({"role": "user", "content": guidance})
                elif next_action == "read_next_link":
                    logger.info("📖 Sufficiency recommends reading another link")
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You have unread promising links in previous search results. "
                                "Use read_url on the next best link before searching again."
                            ),
                        }
                    )

            # ✅ VALIDATION FEEDBACK LOOP (properly nested inside while True)
            eval_result = None
            confidence = "high"

            if self.enable_validation and self.validator and answer_text:
                validation_contexts = memory.get_accumulated_contexts()
                logger.debug(
                    "📋 Validation contexts: %d from memory (%d tokens)",
                    len(validation_contexts),
                    memory.accumulated_tokens,
                )

                if validation_contexts:
                    logger.debug("🔍 Running post-answer validation")
                    eval_result = await self.validator.validate(
                        query=query,
                        response=answer_text,
                        contexts=validation_contexts,
                        session_id=session_id,
                        client=self._client,
                    )
                    logger.debug(
                        "🔍 Validation complete: faith=%.3f halluc=%.3f relevancy=%.3f critical=%s",
                        eval_result.get("faithfulness", -1),
                        eval_result.get("hallucination_rate", -1),
                        eval_result.get("answer_relevancy", -1),
                        eval_result.get("has_critical_failure", False),
                    )

                    if eval_result.get("has_critical_failure"):
                        confidence = "low"
                        logger.warning(
                            "⚠️ Validation flagged critical failure: faith=%.3f halluc=%.3f",
                            eval_result.get("faithfulness", -1),
                            eval_result.get("hallucination_rate", -1),
                        )

                        if validation_retries < _MAX_VALIDATION_RETRIES:
                            validation_retries += 1
                            logger.info(
                                "🔄 Triggering re-search (attempt %d/%d) due to validation failure",
                                validation_retries,
                                _MAX_VALIDATION_RETRIES,
                            )
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        f"Your previous answer failed validation:\n"
                                        f"- Faithfulness: {eval_result.get('faithfulness', 'N/A')}\n"
                                        f"- Hallucination rate: {eval_result.get('hallucination_rate', 'N/A')}\n"
                                        f"- Answer relevancy: {eval_result.get('answer_relevancy', 'N/A')}\n"
                                        f"Search for additional evidence and synthesize again. "
                                        f"Focus on grounding claims in retrieved sources."
                                    ),
                                }
                            )
                            memory.record_synthesis(answer_text)
                            continue
                        else:
                            logger.warning(
                                "⚠️ Max validation retries (%d) exhausted, returning best-effort answer",
                                _MAX_VALIDATION_RETRIES,
                            )
                    else:
                        logger.info(
                            "✅ Validation passed: faith=%.3f halluc=%.3f relevancy=%.3f",
                            eval_result.get("faithfulness", -1),
                            eval_result.get("hallucination_rate", -1),
                            eval_result.get("answer_relevancy", -1),
                        )
                else:
                    logger.warning(
                        "⚠️ No contexts in memory — skipping validation",
                    )
            elif not self.enable_validation:
                logger.debug("⏭️ Validation disabled, skipping")
            elif not answer_text:
                logger.debug("⏭️ Empty answer, skipping validation")

            break

        # --- Post-loop: source extraction and final answer assembly ---
        sources: list[SearchResult] = []
        seen_urls: set[str] = set()
        for step in steps:
            if step.source_url and step.source_url not in seen_urls:
                seen_urls.add(step.source_url)
                sources.append(
                    SearchResult(
                        title=step.source_title or "",
                        url=step.source_url,
                        snippet=step.observation[:200] if step.observation else "",
                        engine="web",
                        score=0.0,
                    )
                )

        final = FinalAnswer(
            answer=answer_text,
            sources=sources,
            steps=steps,
            confidence=confidence,
            total_tokens=total_tokens,
            truncated=truncated,
            eval_result=eval_result,
        )

        status = memory.get_status_summary()
        logger.info(
            "🏁 Search complete: confidence=%s, steps=%d, tokens=%d, truncated=%s, "
            "validated=%s, sources=%d, validation_retries=%d, sufficiency_checks=%d, "
            "memory_tokens=%d/%d, list_items=%d",
            final.confidence,
            len(final.steps),
            final.total_tokens,
            final.truncated,
            final.eval_result is not None,
            len(final.sources),
            validation_retries,
            sufficiency_checks,
            status["tokens_used"],
            status["tokens_max"],
            status["list_items"],
        )

        return final
