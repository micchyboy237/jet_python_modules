"""
ReAct Agent Example with Local Models & Phoenix Tracing.

Demonstrates a retrieval-augmented ReAct loop using local LLM, embedding,
and reranker clients. Includes full OpenTelemetry instrumentation via
custom span helpers for agent, tool, LLM, embedding, and reranker steps.
"""

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests
from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL_LG,
    EMBED_MODEL_LG,
    LLM_BASE_URL,
    LLM_MODEL,
    PHOENIX_REST_API,
    RERANK_BASE_URL,
    RERANK_MODEL,
)
from jet.observability import (
    agent_span,
    console,
    embedding_span,
    get_tracer,
    hash_prompt,
    init_tracing,
    llm_span,
    redact,
    reranker_span,
    tool_span,
)
from openai import OpenAI
from openinference.semconv.trace import (
    DocumentAttributes,
    EmbeddingAttributes,
    RerankerAttributes,
    SpanAttributes,
)
from pydantic import BaseModel, Field
from rich.panel import Panel
from rich.text import Text

# ─── 1. PHOENIX + OTEL SETUP ────────────────────────────────────────────────

PROJECT_NAME = "react-agent-local"

init_tracing(
    project_name=PROJECT_NAME,
    phoenix_rest_api=PHOENIX_REST_API,
)

tracer = get_tracer(__name__)

# ─── 2. CONFIGURATION ───────────────────────────────────────────────────────

PROMPT_TEMPLATE_VERSION = "v3.1"

SYSTEM_PROMPT = """You are a ReAct agent operating in a retrieval-augmented environment. Think step-by-step.

RULES:
1. ALWAYS attempt to use available tools before responding with final_answer. Never assume information is unavailable, outdated, or outside your knowledge without first searching. Your parametric knowledge may be incomplete or stale; the tool-connected knowledge base is authoritative.
2. Use 'thought' to explicitly reason about what you know, what you need, and which tool (if any) can provide it.
3. Use 'action' to call a tool by its exact name, or 'final_answer' ONLY after exhausting relevant tools or confirming no applicable tool exists.
4. Use 'action_input' for tool parameters as a JSON object, or {"answer": "..."} when providing a final answer.
5. If a tool returns insufficient results, refine your query and search again rather than giving up.
6. Do not fabricate, speculate, or extrapolate beyond what tools return. If tools yield no relevant information after reasonable attempts, state that clearly in final_answer.

AVAILABLE TOOLS:
- search_docs: Retrieve and rerank documents from the knowledge base. Requires {"query": "<search string>"}.

RESPONSE FORMAT:
You MUST respond with valid JSON matching this exact schema:
{
  "thought": "<your reasoning>",
  "action": "<tool_name | final_answer>",
  "action_input": {<tool params> | {"answer": "<response>"}}
}"""

TOOL_SCHEMA_VERSION = "v1.2"

# ─── 3. STRUCTURED OUTPUT SCHEMA ────────────────────────────────────────────


class AgentAction(BaseModel):
    thought: str = Field(description="Step-by-step reasoning")
    action: str = Field(description="Tool name or 'final_answer'")
    action_input: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool parameters or final answer payload",
    )


AGENT_ACTION_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "AgentAction",
        "strict": True,
        "schema": AgentAction.model_json_schema(),
    },
}


# ─── 4. LOCAL CLIENT WRAPPERS WITH REUSABLE OBSERVABILITY ───────────────────


class LocalLLMClient:
    def __init__(self, base_url: str, model_name: str):
        self.client = OpenAI(base_url=base_url.rstrip("/"), api_key="local")
        self.model_name = model_name

    def chat(self, messages: list[dict], response_format=None, **kwargs) -> dict:
        invocation_params: dict[str, Any] = {
            "max_tokens": kwargs.get("max_tokens", 8192),
            "temperature": kwargs.get("temperature", 0.3),
            "top_p": kwargs.get("top_p", 0.95),
            "presence_penalty": kwargs.get("presence_penalty", 1.5),
        }

        with llm_span(
            name="llm.chat",
            model_name=self.model_name,
            messages=messages,
            invocation_params=invocation_params,
            provider="llama_cpp",
        ) as span:
            create_kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": invocation_params["max_tokens"],
                "temperature": invocation_params["temperature"],
                "top_p": invocation_params["top_p"],
                "presence_penalty": invocation_params["presence_penalty"],
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            }

            if response_format is not None:
                create_kwargs["response_format"] = response_format

            stream = self.client.chat.completions.create(**create_kwargs)

            collected_content: list[str] = []
            usage_data: dict[str, int] = {}

            console.print(Text("🤖 LLM: ", style="bold cyan"), end="")

            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None

                if delta and delta.content:
                    collected_content.append(delta.content)
                    print(delta.content, end="", flush=True)

                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = {
                        "prompt_tokens": chunk.usage.prompt_tokens or 0,
                        "completion_tokens": chunk.usage.completion_tokens or 0,
                        "total_tokens": chunk.usage.total_tokens or 0,
                    }

            print(flush=True)

            output = "".join(collected_content)

            span.set_attribute(
                SpanAttributes.LLM_OUTPUT_MESSAGES,
                json.dumps(
                    [{"role": "assistant", "content": redact(output)}],
                    ensure_ascii=False,
                ),
            )
            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
                usage_data.get("prompt_tokens", 0),
            )
            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                usage_data.get("completion_tokens", 0),
            )
            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                usage_data.get("total_tokens", 0),
            )

            return {"content": output, "usage": usage_data}


class LocalEmbedderClient:
    def __init__(self, base_url: str, model_name: str):
        self.client = OpenAI(base_url=base_url.rstrip("/"), api_key="local")
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        with embedding_span(
            name="CreateEmbeddings",
            model_name=self.model_name,
            texts=texts,
        ) as span:
            resp = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
            )

            embeddings = [item.embedding for item in resp.data]

            for i, (text, vector) in enumerate(zip(texts, embeddings)):
                span.set_attribute(
                    f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{i}.{EmbeddingAttributes.EMBEDDING_TEXT}",
                    redact(text),
                )
                span.set_attribute(
                    f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{i}.{EmbeddingAttributes.EMBEDDING_VECTOR}",
                    vector,
                )

            if hasattr(resp, "usage") and resp.usage:
                span.set_attribute(
                    SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
                    resp.usage.prompt_tokens or 0,
                )
                span.set_attribute(
                    SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                    resp.usage.total_tokens or 0,
                )

            return embeddings


class LocalRerankerClient:
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[dict]:
        console.print(f"[dim]🔎 Reranker INPUT docs count: {len(documents)}[/dim]")

        with reranker_span(
            name="reranker.rerank",
            model_name=self.model_name,
            query=query,
            documents=documents,
            top_k=top_k,
        ) as span:
            console.print(
                f"[green]✅ Set {len(documents)} indexed input_documents attributes[/green]"
            )

            resp = requests.post(
                f"{self.base_url}/rerank",
                json={
                    "model": self.model_name,
                    "query": query,
                    "documents": documents,
                    "top_k": top_k,
                },
                timeout=30,
            ).json()

            raw_results = resp.get("results", [])

            enriched_results: list[dict] = []
            for r in raw_results:
                idx = r["index"]
                enriched_results.append(
                    {
                        "document": documents[idx],
                        "relevance_score": r["relevance_score"],
                        "index": idx,
                    }
                )

            for i, result in enumerate(enriched_results):
                span.set_attribute(
                    f"{RerankerAttributes.RERANKER_OUTPUT_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_CONTENT}",
                    redact(result["document"][:2000]),
                )
                span.set_attribute(
                    f"{RerankerAttributes.RERANKER_OUTPUT_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_SCORE}",
                    float(result["relevance_score"]),
                )

            console.print(
                f"[dim]🔎 Reranker OUTPUT: {len(enriched_results)} docs returned[/dim]"
            )

            return enriched_results


# ─── 5. TOOL REGISTRY WITH REUSABLE OBSERVABILITY ───────────────────────────


def search_docs(
    query: str,
    embedder: LocalEmbedderClient,
    reranker: LocalRerankerClient,
) -> str:
    with tool_span(
        name="tool.search_docs",
        tool_name="search_docs",
        parameters={"query": query},
        schema_version=TOOL_SCHEMA_VERSION,
    ) as span:
        console.print(
            Panel(
                f"🔍 Searching docs for: {redact(query)}",
                title="Tool Execution",
                border_style="yellow",
            )
        )

        embedder.embed([query])

        candidate_docs = [
            "Q3 2026 Earnings Report: Revenue grew 12% YoY to $4.2B driven by AI product adoption. Operating margin expanded to 28%. EPS of $3.15 beat consensus by 8%.",
            "Q2 2026 Earnings Report: Revenue of $3.8B, up 9% YoY. Cloud segment grew 22%. Company raised full-year guidance citing strong enterprise demand.",
            "Annual Report FY2025: Full-year revenue $14.1B. R&D spending increased 18% focused on generative AI infrastructure. Share buyback program expanded to $5B.",
        ]

        reranked = reranker.rerank(query, candidate_docs, top_k=3)

        result = "\n---\n".join([r["document"] for r in reranked])

        span.set_attribute(SpanAttributes.OUTPUT_VALUE, redact(result[:2000]))
        span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "text/plain")
        span.set_attribute("tool.output_full_length", len(result))

        console.print(
            Text(f"✅ Found {len(reranked)} relevant documents", style="green")
        )

        return result


TOOLS = {"search_docs": search_docs}


# ─── 6. REACT LOOP WITH COMPLETE HIERARCHICAL OBSERVABILITY ─────────────────


@dataclass
class LoopMeta:
    total_steps: int = 0
    total_tokens: int = 0
    repeated_tool_calls: int = 0
    success: bool = False
    failure_reason: str | None = None
    previous_tool_signatures: list[str] = field(default_factory=list)


def run_react_loop(
    user_query: str,
    llm: LocalLLMClient,
    embedder: LocalEmbedderClient,
    reranker: LocalRerankerClient,
    max_steps: int = 10,
) -> str:
    session_id = str(uuid.uuid4())
    meta = LoopMeta()
    final_answer = "No answer produced"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    console.print(Panel(user_query, title="🎯 User Query", border_style="bold blue"))

    with agent_span(
        name="react_agent.session",
        session_id=session_id,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        system_prompt_hash=hash_prompt(SYSTEM_PROMPT),
        max_steps=max_steps,
    ) as root_span:
        # Attach user query to root span for top-level visibility
        root_span.set_attribute(SpanAttributes.INPUT_VALUE, redact(user_query))
        root_span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "text/plain")

        for step in range(max_steps):
            meta.total_steps += 1
            console.rule(f"[bold magenta]Step {step + 1}/{max_steps}")

            with tracer.start_as_current_span(
                f"react_loop.iteration_{step}",
                attributes={
                    "react.step_number": step,
                    "react.prompt_version": PROMPT_TEMPLATE_VERSION,
                    "react.message_count": len(messages),
                },
            ) as iter_span:
                # ── LLM Call (creates child llm.chat span via reusable builder) ──
                response = llm.chat(
                    messages,
                    response_format=AGENT_ACTION_JSON_SCHEMA,
                )

                raw_output = response["content"]
                step_tokens = response["usage"].get("total_tokens", 0)
                meta.total_tokens += step_tokens

                iter_span.set_attribute("react.step_tokens", step_tokens)

                # ── Parse & Validate Structured Output ──
                try:
                    parsed = AgentAction.model_validate_json(raw_output)
                except Exception as e:
                    iter_span.record_exception(e)
                    iter_span.set_attribute("react.parse_error", str(e)[:1000])
                    iter_span.set_attribute(
                        "react.raw_output", redact(raw_output[:2000])
                    )

                    observation = (
                        "System parsing error: Your previous response was not valid "
                        "JSON matching the required schema. Retry with valid JSON only."
                    )

                    messages.append({"role": "assistant", "content": raw_output})
                    messages.append({"role": "user", "content": observation})

                    console.print(
                        Text(
                            f"❌ Failed to parse agent JSON: {type(e).__name__}: {str(e)[:300]}",
                            style="bold red",
                        )
                    )
                    continue

                thought = parsed.thought
                action = parsed.action
                action_input = parsed.action_input

                # ── Business-Specific Iteration Attributes ──
                iter_span.set_attribute("react.thought_raw", redact(thought))
                iter_span.set_attribute("react.planned_action", action)
                iter_span.set_attribute(
                    "react.planned_action_input",
                    json.dumps(
                        {k: redact(str(v)) for k, v in action_input.items()},
                        ensure_ascii=False,
                    ),
                )

                console.print(Text(f"\n💭 Thought: {thought}", style="italic dim"))
                console.print(Text(f"⚡ Action: {action}", style="bold yellow"))

                # ── Terminal Condition: Final Answer ──
                if action == "final_answer":
                    meta.success = True
                    final_answer = action_input.get("answer", "")

                    root_span.set_attribute(
                        "agent.final_answer",
                        redact(final_answer[:3000]),
                    )
                    root_span.set_attribute(
                        SpanAttributes.OUTPUT_VALUE, redact(final_answer[:3000])
                    )
                    root_span.set_attribute(
                        SpanAttributes.OUTPUT_MIME_TYPE, "text/plain"
                    )
                    iter_span.set_attribute("react.final_answer_reached", True)
                    iter_span.set_attribute(
                        "react.final_answer", redact(final_answer[:3000])
                    )

                    console.print(
                        Panel(
                            final_answer,
                            title="✅ Final Answer",
                            border_style="bold green",
                        )
                    )
                    break

                # ── Unknown Tool Error ──
                if action not in TOOLS:
                    iter_span.set_attribute("react.error", f"Unknown tool: {action}")
                    iter_span.set_attribute("react.error_type", "unknown_tool")

                    messages.append({"role": "assistant", "content": raw_output})

                    error_msg = (
                        f"Error: Unknown tool '{action}'. "
                        f"Available: {list(TOOLS.keys())}"
                    )
                    messages.append({"role": "user", "content": error_msg})

                    console.print(Text(f"❌ {error_msg}", style="bold red"))
                    continue

                # ── Repeated Tool Call Detection & Circuit Breaker ──
                tool_sig = f"{action}:{json.dumps(action_input, sort_keys=True)}"

                if tool_sig in meta.previous_tool_signatures:
                    meta.repeated_tool_calls += 1
                    repeat_count = meta.previous_tool_signatures.count(tool_sig)

                    iter_span.set_attribute("react.repeated_call", True)
                    iter_span.set_attribute("react.repeat_count", repeat_count)
                    iter_span.set_attribute("react.tool_signature", tool_sig)

                    console.print(Text("⚠️ Repeated tool call detected", style="yellow"))

                    if repeat_count >= 2:
                        iter_span.set_attribute("react.circuit_breaker_triggered", True)

                        console.print(
                            Text(
                                "🛑 Circuit breaker: forcing final_answer after repeated failures",
                                style="bold red",
                            )
                        )

                        messages.append({"role": "assistant", "content": raw_output})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "SYSTEM: You have called the same tool with the same "
                                    "parameters multiple times without progress. STOP "
                                    "searching. Provide your best answer based on the "
                                    "observations you already have, or clearly state that "
                                    "the information is unavailable."
                                ),
                            }
                        )
                        continue

                meta.previous_tool_signatures.append(tool_sig)

                # ── Tool Execution (creates child tool.* span via reusable builder) ──
                try:
                    observation = TOOLS[action](
                        **action_input,
                        embedder=embedder,
                        reranker=reranker,
                    )
                    iter_span.set_attribute(
                        "react.observation_length", len(observation)
                    )
                    iter_span.set_attribute(
                        "react.observation_preview",
                        redact(observation[:500]),
                    )

                except Exception as e:
                    observation = (
                        f"Tool execution error: {type(e).__name__}: {str(e)[:500]}"
                    )

                    iter_span.set_attribute("react.tool_error", str(e)[:1000])
                    iter_span.set_attribute("react.tool_error_type", type(e).__name__)
                    iter_span.record_exception(e)

                    console.print(
                        Text(f"💥 Tool Error: {observation}", style="bold red")
                    )

                messages.append({"role": "assistant", "content": raw_output})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Observation: {observation}",
                    }
                )

        else:
            meta.failure_reason = "max_steps_exhausted"
            root_span.set_attribute("agent.loop.failure_reason", meta.failure_reason)

            console.print(
                Text(
                    "⏰ Max steps exhausted without final answer",
                    style="bold red",
                )
            )

        # ── Aggregate Loop Metrics on Root Span ──
        root_span.set_attribute("agent.loop.total_steps", meta.total_steps)
        root_span.set_attribute("agent.loop.total_tokens", meta.total_tokens)
        root_span.set_attribute(
            "agent.loop.repeated_tool_calls",
            meta.repeated_tool_calls,
        )
        root_span.set_attribute("agent.loop.success", meta.success)
        root_span.set_attribute(
            "agent.loop.unique_tool_signatures",
            len(set(meta.previous_tool_signatures)),
        )
        root_span.set_attribute(
            "agent.loop.total_tool_calls",
            len(meta.previous_tool_signatures),
        )

        if meta.failure_reason:
            root_span.set_attribute(
                "agent.loop.failure_reason",
                meta.failure_reason,
            )

        console.rule("[bold]Session Summary")
        console.print(
            f"Steps: {meta.total_steps} | Tokens: {meta.total_tokens} | "
            f"Repeated Calls: {meta.repeated_tool_calls} | Success: {meta.success}"
        )

        # ── Trace Link ──
        phoenix_host = PHOENIX_REST_API.rstrip("/")
        if phoenix_host.endswith("/v1"):
            phoenix_host = phoenix_host[:-3]

        trace_id_hex = format(root_span.get_span_context().trace_id, "032x")
        trace_url = f"{phoenix_host}/redirects/traces/{trace_id_hex}"

        console.rule("[bold]🔗 Trace Link")
        console.print(f"[link={trace_url}]{trace_url}[/link]", style="bold cyan")
        console.print(f"[dim]Session ID: {session_id}[/dim]")
        console.print(f"[dim]Trace ID:   {trace_id_hex}[/dim]")

        return final_answer


# ─── 7. USAGE ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the ReAct Agent with a user query."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="What were the key findings in the Q3 2026 earnings report?",
        help=(
            "User query to run. Default: "
            '"What were the key findings in the Q3 2026 earnings report?"'
        ),
    )

    args = parser.parse_args()

    llm = LocalLLMClient(LLM_BASE_URL, LLM_MODEL)
    embedder = LocalEmbedderClient(EMBED_BASE_URL_LG, EMBED_MODEL_LG)
    reranker = LocalRerankerClient(RERANK_BASE_URL, RERANK_MODEL)

    answer = run_react_loop(
        user_query=args.query,
        llm=llm,
        embedder=embedder,
        reranker=reranker,
    )
