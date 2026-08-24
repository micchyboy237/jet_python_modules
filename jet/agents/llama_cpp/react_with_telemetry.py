import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL_LG,
    EMBED_MODEL_LG,
    LLM_BASE_URL,
    LLM_MODEL,
    PHOENIX_REST_API,
    RERANK_BASE_URL,
    RERANK_MODEL,
)
from openai import OpenAI
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from phoenix.otel import register
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# ─── 1. PHOENIX + OTEL SETUP ────────────────────────────────────────────────
register(
    project_name="react-agent-local",
    endpoint=f"{PHOENIX_REST_API}/traces",
)
tracer = trace.get_tracer(__name__)
console = Console(force_terminal=True, highlight=False)

# ─── 2. CONFIGURATION & REDACTION ────────────────────────────────────────────
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

PII_PATTERNS = ["ssn", "password", "api_key", "secret", "token"]


def redact(text: str) -> str:
    lower = text.lower()
    for pattern in PII_PATTERNS:
        if pattern in lower:
            return "[REDACTED: contains sensitive content]"
    return text


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


# ─── 3. STRUCTURED OUTPUT SCHEMA ────────────────────────────────────────────
class AgentAction(BaseModel):
    thought: str = Field(description="Step-by-step reasoning")
    action: str = Field(description="Tool name or 'final_answer'")
    action_input: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool parameters or final answer payload",
    )


# Pre-compute JSON schema once for reuse across all LLM calls
AGENT_ACTION_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "AgentAction",
        "strict": True,
        "schema": AgentAction.model_json_schema(),
    },
}


# ─── 4. LLAMA.CPP CLIENT WRAPPERS WITH INSTRUMENTATION ──────────────────────
class LocalLLMClient:
    def __init__(self, base_url: str, model_name: str):
        self.client = OpenAI(base_url=base_url.rstrip("/"), api_key="local")
        self.model_name = model_name

    def chat(self, messages: list[dict], response_format=None, **kwargs) -> dict:
        with tracer.start_as_current_span(
            "llm.chat",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                SpanAttributes.LLM_MODEL_NAME: self.model_name,
                SpanAttributes.LLM_PROVIDER: "llama_cpp",
                SpanAttributes.LLM_INVOCATION_PARAMETERS: json.dumps(kwargs),
            },
        ) as span:
            safe_messages = [
                {"role": m["role"], "content": redact(m["content"])} for m in messages
            ]
            span.set_attribute(
                SpanAttributes.LLM_INPUT_MESSAGES, json.dumps(safe_messages)
            )

            create_kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": kwargs.get("max_tokens", 8192),
                "temperature": kwargs.get("temperature", 0.3),
                "top_p": kwargs.get("top_p", 0.95),
                "presence_penalty": kwargs.get("presence_penalty", 1.5),
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            }

            # Pass JSON schema dict (not Pydantic class) to preserve streaming
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
                json.dumps([{"role": "assistant", "content": redact(output)}]),
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
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL, usage_data.get("total_tokens", 0)
            )

            return {"content": output, "usage": usage_data}


class LocalEmbedderClient:
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        with tracer.start_as_current_span(
            "embedder.embed",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.EMBEDDING.value,
                SpanAttributes.EMBEDDING_MODEL_NAME: self.model_name,
                SpanAttributes.EMBEDDING_TEXTS: json.dumps([redact(t) for t in texts]),
            },
        ):
            resp = OpenAI(base_url=self.base_url, api_key="local").embeddings.create(
                model=self.model_name, input=texts
            )
            return [item.embedding for item in resp.data]


class LocalRerankerClient:
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[dict]:
        with tracer.start_as_current_span(
            "reranker.rerank",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RETRIEVER.value,
                SpanAttributes.RETRIEVAL_QUERY_TEXT: redact(query),
                "reranker.model_name": self.model_name,
                "reranker.top_k": top_k,
            },
        ) as span:
            import requests

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
            results = resp.get("results", [])
            span.set_attribute("reranker.result_count", len(results))
            return results


# ─── 5. TOOL REGISTRY WITH INSTRUMENTATION ───────────────────────────────────
TOOL_SCHEMA_VERSION = "v1.2"


def search_docs(
    query: str, embedder: LocalEmbedderClient, reranker: LocalRerankerClient
) -> str:
    with tracer.start_as_current_span(
        "tool.search_docs",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
            SpanAttributes.TOOL_NAME: "search_docs",
            SpanAttributes.TOOL_PARAMETERS: json.dumps({"query": redact(query)}),
            "tool.schema_version": TOOL_SCHEMA_VERSION,
        },
    ) as span:
        console.print(
            Panel(
                f"🔍 Searching docs for: {redact(query)}",
                title="Tool Execution",
                border_style="yellow",
            )
        )

        embeddings = embedder.embed([query])
        candidate_docs = ["doc1 content", "doc2 content", "doc3 content"]
        reranked = reranker.rerank(query, candidate_docs, top_k=3)

        result = "\n---\n".join([r["document"] for r in reranked])
        span.set_attribute(SpanAttributes.TOOL_OUTPUT, redact(result[:2000]))
        span.set_attribute("tool.output_full_length", len(result))

        console.print(
            Text(f"✅ Found {len(reranked)} relevant documents", style="green")
        )
        return result


TOOLS = {"search_docs": search_docs}


# ─── 6. REACT LOOP WITH HIERARCHICAL TRACING ─────────────────────────────────
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
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    console.print(Panel(user_query, title="🎯 User Query", border_style="bold blue"))

    with tracer.start_as_current_span(
        "react_agent.session",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
            SpanAttributes.SESSION_ID: session_id,
            "agent.prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "agent.system_prompt_hash": hash_prompt(SYSTEM_PROMPT),
            "agent.max_steps": max_steps,
        },
    ) as root_span:
        for step in range(max_steps):
            meta.total_steps += 1
            console.rule(f"[bold magenta]Step {step + 1}/{max_steps}")

            with tracer.start_as_current_span(
                f"react_loop.iteration_{step}",
                attributes={"react.step_number": step},
            ) as iter_span:
                with tracer.start_as_current_span(
                    "react.thought_generation",
                    attributes={
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                        "react.prompt_version": PROMPT_TEMPLATE_VERSION,
                    },
                ) as thought_span:
                    # Pass pre-computed JSON schema dict, NOT the Pydantic class
                    response = llm.chat(
                        messages, response_format=AGENT_ACTION_JSON_SCHEMA
                    )
                    raw_output = response["content"]
                    meta.total_tokens += response["usage"].get("total_tokens", 0)

                    # Validate against Pydantic model after streaming completes
                    parsed = AgentAction.model_validate_json(raw_output)
                    thought = parsed.thought
                    action = parsed.action
                    action_input = parsed.action_input

                    thought_span.set_attribute("react.thought_raw", redact(thought))
                    thought_span.set_attribute("react.planned_action", action)
                    thought_span.set_attribute(
                        "react.planned_action_input",
                        json.dumps(
                            {k: redact(str(v)) for k, v in action_input.items()}
                        ),
                    )

                    console.print(Text(f"\n💭 Thought: {thought}", style="italic dim"))
                    console.print(Text(f"⚡ Action: {action}", style="bold yellow"))

                if action == "final_answer":
                    meta.success = True
                    final_answer = action_input.get("answer", "")
                    root_span.set_attribute(
                        "agent.final_answer", redact(final_answer[:3000])
                    )
                    console.print(
                        Panel(
                            final_answer,
                            title="✅ Final Answer",
                            border_style="bold green",
                        )
                    )
                    break

                if action not in TOOLS:
                    iter_span.set_attribute("react.error", f"Unknown tool: {action}")
                    messages.append({"role": "assistant", "content": raw_output})
                    error_msg = f"Error: Unknown tool '{action}'. Available: {list(TOOLS.keys())}"
                    messages.append({"role": "user", "content": error_msg})
                    console.print(Text(f"❌ {error_msg}", style="bold red"))
                    continue

                tool_sig = f"{action}:{json.dumps(action_input, sort_keys=True)}"
                if tool_sig in meta.previous_tool_signatures:
                    meta.repeated_tool_calls += 1
                    iter_span.set_attribute("react.repeated_call", True)
                    console.print(Text("⚠️ Repeated tool call detected", style="yellow"))
                meta.previous_tool_signatures.append(tool_sig)

                try:
                    observation = TOOLS[action](
                        **action_input, embedder=embedder, reranker=reranker
                    )
                    iter_span.set_attribute(
                        "react.observation_length", len(observation)
                    )
                except Exception as e:
                    observation = (
                        f"Tool execution error: {type(e).__name__}: {str(e)[:500]}"
                    )
                    iter_span.set_attribute("react.tool_error", str(e)[:1000])
                    iter_span.record_exception(e)
                    console.print(
                        Text(f"💥 Tool Error: {observation}", style="bold red")
                    )

                messages.append({"role": "assistant", "content": raw_output})
                messages.append(
                    {"role": "user", "content": f"Observation: {observation}"}
                )

        else:
            meta.failure_reason = "max_steps_exhausted"
            console.print(
                Text("⏰ Max steps exhausted without final answer", style="bold red")
            )

        root_span.set_attribute("agent.loop.total_steps", meta.total_steps)
        root_span.set_attribute("agent.loop.total_tokens", meta.total_tokens)
        root_span.set_attribute(
            "agent.loop.repeated_tool_calls", meta.repeated_tool_calls
        )
        root_span.set_attribute("agent.loop.success", meta.success)
        if meta.failure_reason:
            root_span.set_attribute("agent.loop.failure_reason", meta.failure_reason)

        console.rule("[bold]Session Summary")
        console.print(
            f"Steps: {meta.total_steps} | Tokens: {meta.total_tokens} | "
            f"Repeated Calls: {meta.repeated_tool_calls} | Success: {meta.success}"
        )

        return root_span.attributes.get("agent.final_answer", "No answer produced")


# ─── 7. USAGE ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    llm = LocalLLMClient(LLM_BASE_URL, LLM_MODEL)
    embedder = LocalEmbedderClient(EMBED_BASE_URL_LG, EMBED_MODEL_LG)
    reranker = LocalRerankerClient(RERANK_BASE_URL, RERANK_MODEL)

    answer = run_react_loop(
        user_query="What were the key findings in the Q3 2026 earnings report?",
        llm=llm,
        embedder=embedder,
        reranker=reranker,
    )
