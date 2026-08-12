"""
WebSwarm RAG Agent - Autonomous Multi-Page Research Pipeline
Features:
- Starts from a root URL and crawls linked pages to gather knowledge
- Embeds and reranks scraped content against the user query
- Uses an LLM evaluator to determine if enough information has been collected
- Iteratively fetches, scores, and synthesizes until the answer is sufficient
- Supports cross-domain crawling, sigmoid score normalization, and token-safe truncation
- Streams LLM output in real-time with structured JSON evaluation
Use Cases:
- Researching documentation, product specs, or technical topics across a website
- Automated FAQ answering from official docs or knowledge bases
- Competitive research by crawling multiple product pages
- Summarizing scattered information from a domain into a single coherent answer
Usage Examples:
    # Run a research query against LangGraph documentation
    asyncio.run(run_webswarm(
        query="What are the deployment options for LangGraph Platform?",
        root_url="https://langchain-ai.github.io/langgraph/"
    ))
    # Customize behavior via module-level constants:
    # MAX_ITERATIONS = 10          # increase search depth
    # RELEVANCE_THRESHOLD = 0.25   # stricter relevance filtering
    # FOLLOW_CROSS_DOMAIN_LINKS = False  # stay on the same domain
    # APPLY_SIGMOID_NORMALIZATION = False  # use raw reranker scores
"""

import asyncio
import json
from typing import Annotated, Literal, Sequence, TypedDict
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from jet.adapters.llama_cpp.chunking_utils import truncate_texts
from jet.adapters.llama_cpp.config import (
    LLM_BASE_URL,
    LLM_MODEL,
    RERANK_BASE_URL,
    RERANK_MODEL,
)
from jet.adapters.llama_cpp.llm_utils import chat
from jet.adapters.llama_cpp.rerank_utils import rerank
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

MAX_ITERATIONS = 5
MAX_PAGES_PER_ITER = 3
RELEVANCE_THRESHOLD = 0.15
MIN_CHUNKS_TO_KEEP = 3
APPLY_SIGMOID_NORMALIZATION = True
FOLLOW_CROSS_DOMAIN_LINKS = True
MIN_SCORE_FOR_KB = 0.05
LLM_MAX_TOKENS = 2048
RERANK_MAX_TOKENS = 1024
RERANK_QUERY_TOKEN_RESERVE = 200


class KBEntry(TypedDict):
    """A single knowledge base entry with relevance metadata."""

    url: str
    content: str
    score: float
    raw_score: float
    original_chars: int
    truncated_chars: int


class FetchMeta(TypedDict):
    """Metadata from fetching and parsing a single URL."""

    url: str
    status: int | None
    final_url: str | None
    text_len: int
    links_found: int
    links_filtered: int
    error: str | None


class EvalResult(TypedDict):
    """Structured evaluation response from the LLM."""

    evaluation: Literal["sufficient", "insufficient", "irrelevant"]
    reasoning: str


class WebSwarmState(TypedDict):
    """LangGraph state for the WebSwarm RAG pipeline."""

    query: str
    root_url: str
    messages: Annotated[Sequence, add_messages]
    visited_urls: set[str]
    pending_urls: list[str]
    knowledge_base: list[KBEntry]
    iteration: int
    evaluation: Literal["sufficient", "insufficient", "irrelevant"]
    final_answer: str | None


class RetrieveNodeOutput(TypedDict):
    """Partial state update returned by retrieve_node."""

    knowledge_base: list[KBEntry]
    visited_urls: set[str]
    pending_urls: list[str]
    iteration: int


class EvaluateNodeOutput(TypedDict):
    """Partial state update returned by evaluate_node."""

    evaluation: Literal["sufficient", "insufficient", "irrelevant"]


class SynthesizeNodeOutput(TypedDict):
    """Partial state update returned by synthesize_node."""

    final_answer: str


class RetrieveStep(TypedDict):
    """Captured input/output for a single retrieve_node execution."""

    iteration: int
    pending_urls: list[str]
    visited_count: int
    fetched_pages: int
    discovered_links: int
    reranked_chunks: list[dict]
    chunks_above_threshold: int
    chunks_added_to_kb: int
    kb_total_after: int
    next_pending_count: int


class EvaluateStep(TypedDict):
    """Captured input/output for a single evaluate_node execution."""

    iteration: int
    kb_size: int
    pending_available: bool
    pending_count: int
    evaluation: Literal["sufficient", "insufficient", "irrelevant"]
    reasoning: str


class SynthesizeStep(TypedDict):
    """Captured input/output for synthesize_node execution."""

    kb_entries_used: int
    source_urls: list[str]
    source_scores: list[float]
    answer: str


class RouteDecision(TypedDict):
    """Captured routing decision from should_continue."""

    iteration: int
    decision: Literal["retrieve", "synthesize"]
    reason: str


class SwarmResult(TypedDict):
    """Complete output of the WebSwarm pipeline with node-level trace data."""

    answer: str | None
    sources: list[str]
    iterations: int
    pages_visited: int
    knowledge_base: list[KBEntry]
    config: dict
    retrieve_steps: list[RetrieveStep]
    evaluate_steps: list[EvaluateStep]
    synthesize_step: SynthesizeStep | None
    route_decisions: list[RouteDecision]


def prepare_docs_for_rerank(
    chunks: list[KBEntry],
    max_tokens: int = RERANK_MAX_TOKENS,
    query_reserve: int = RERANK_QUERY_TOKEN_RESERVE,
) -> list[KBEntry]:
    """Truncate chunk content to fit within reranker's physical batch size.

    Uses the reranker model's own tokenizer via jet's truncate_texts,
    which respects sentence boundaries to avoid cutting mid-thought.
    """
    if not chunks:
        return chunks

    effective_max = max(64, max_tokens - query_reserve)
    contents = [c["content"] for c in chunks]
    truncated = truncate_texts(
        texts=contents,
        model=RERANK_MODEL,
        max_tokens=effective_max,
        strict_sentences=True,
        show_progress=False,
    )

    for chunk, trunc_text in zip(chunks, truncated):
        original_len = len(chunk["content"])
        chunk["content"] = trunc_text
        chunk["original_chars"] = original_len
        chunk["truncated_chars"] = len(trunc_text)

    truncated_count = sum(
        1 for c in chunks if c["original_chars"] != c["truncated_chars"]
    )
    if truncated_count > 0:
        print(
            f"[PREP] Truncated {truncated_count}/{len(chunks)} docs to fit "
            f"reranker limit ({effective_max} tokens, model={RERANK_MODEL})"
        )

    return chunks


def rerank_chunks(query: str, chunks: list[KBEntry]) -> list[KBEntry]:
    """Rerank chunks using jet's rerank_utils.rerank().

    Delegates to the shared rerank utility which handles:
    - HTTP request to the reranker endpoint
    - Score normalization via sigmoid (configurable)
    - Typed result sorting by relevance

    Args:
        query: Search query string
        chunks: List of KBEntry dicts with 'content' field

    Returns:
        Same chunks list sorted by reranker score (descending),
        with 'score' and 'raw_score' fields populated.
    """
    if not chunks:
        print("[RERANK] ⚠️  No chunks to rerank")
        return []

    texts = [c["content"] for c in chunks]
    print(
        f"[RERANK] Sending {len(texts)} docs to reranker "
        f"(model={RERANK_MODEL}, normalize={APPLY_SIGMOID_NORMALIZATION})"
    )

    results = rerank(
        query=query,
        documents=texts,
        top_n=len(texts),
        normalize_scores=APPLY_SIGMOID_NORMALIZATION,
    )

    # Map results back to original chunks
    score_map = {r["index"]: r for r in results}
    for i, chunk in enumerate(chunks):
        if i in score_map:
            chunk["score"] = score_map[i]["score"]
            chunk["raw_score"] = score_map[i]["raw_score"]
        else:
            chunk["score"] = 0.0
            chunk["raw_score"] = 0.0

    ranked = sorted(chunks, key=lambda x: x["score"], reverse=True)

    if ranked:
        print(
            f"[RERANK] Score distribution "
            f"({'normalized' if APPLY_SIGMOID_NORMALIZATION else 'raw'}):"
        )
        for i, c in enumerate(ranked):
            content_preview = c["content"][:80].replace("\n", " ")
            orig = c.get("original_chars", "?")
            trunc = c.get("truncated_chars", "?")
            trunc_info = f" chars({orig}→{trunc})" if orig != trunc else ""
            print(
                f"  [{i}] score={c['score']:.4f} raw={c.get('raw_score', 'N/A')} "
                f'url={c["url"]}{trunc_info} preview="{content_preview}..."'
            )
    else:
        print("[RERANK] ⚠️  No ranked results returned")

    return ranked


async def fetch_and_parse(url: str) -> tuple[str, list[str], FetchMeta]:
    """Fetch and parse a single URL, extracting text and links.

    Args:
        url: The URL to fetch

    Returns:
        Tuple of (text_content, discovered_links, fetch_metadata)
    """
    meta: FetchMeta = {
        "url": url,
        "status": None,
        "final_url": None,
        "text_len": 0,
        "links_found": 0,
        "links_filtered": 0,
        "error": None,
    }

    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "WebSwarmBot/1.0"})
            resp.raise_for_status()
            meta["status"] = resp.status_code
            meta["final_url"] = str(resp.url)

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)[:8000]
        meta["text_len"] = len(text)

        if meta["final_url"] != url:
            print(f"[FETCH] ↪ Redirect detected: {url} → {meta['final_url']}")

        base_domain = urlparse(url).netloc
        final_domain = urlparse(meta["final_url"]).netloc

        all_links: list[str] = []
        for a in soup.find_all("a", href=True):
            full_url = urljoin(str(resp.url), a["href"])
            parsed = urlparse(full_url)
            if parsed.scheme in ("http", "https") and full_url not in all_links:
                all_links.append(full_url)
        meta["links_found"] = len(all_links)

        if FOLLOW_CROSS_DOMAIN_LINKS:
            filtered = [
                l
                for l in all_links
                if not any(
                    skip in l.lower()
                    for skip in ["#", "javascript:", "mailto:", ".pdf", ".zip"]
                )
            ]
        else:
            filtered = [l for l in all_links if urlparse(l).netloc == base_domain]
        meta["links_filtered"] = len(filtered)
        links = filtered[:20]

        print(
            f"[FETCH] ✅ {url} | status={meta['status']} | "
            f"text={meta['text_len']} chars | links={meta['links_found']} found, "
            f"{meta['links_filtered']} after filter | final_domain={final_domain}"
        )
        return text, links, meta

    except Exception as e:
        meta["error"] = str(e)
        print(f"[FETCH] ❌ {url} | error={e}")
        return "", [], meta


def llm_completion(
    prompt: str,
    max_tokens: int = LLM_MAX_TOKENS,
    temperature: float = 0.3,
    response_format: dict | None = None,
    label: str = "LLM",
) -> str:
    """Call LLM via jet's llm_utils.chat() which wraps the full streaming pipeline.

    Uses the shared chat utility which provides:
    - Streaming output with real-time token display
    - Phoenix observability (when configured)
    - Configurable sampling parameters
    - Structured output support via response_format
    - Tool calling support (unused here but available)

    Args:
        prompt: The prompt text to send
        max_tokens: Maximum completion tokens
        temperature: Sampling temperature (0.0 for deterministic)
        response_format: Optional response format (e.g., {"type": "json_object"})
        label: Label for log output identification

    Returns:
        Complete LLM response as string
    """
    prompt_chars = len(prompt)
    print(
        f"\n[{label}] Starting generation (prompt={prompt_chars} chars, "
        f"max_tokens={max_tokens}, temp={temperature})",
        flush=True,
    )

    print(f"[{label}] ", end="", flush=True)
    result = chat(
        prompt=prompt,
        model=LLM_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        presence_penalty=1.5,
        response_format=response_format,
        enable_thinking=False,
        project_name="rag_web_crawler_agent",
    )

    content = result.content
    print(f"[{label}] ", end="", flush=True)
    print(content, flush=True)
    print(
        f"[{label}] Completed: {len(content)} chars, "
        f"finish_reason={result.finish_reason}",
        flush=True,
    )

    return content


async def retrieve_node(state: WebSwarmState) -> RetrieveNodeOutput:
    """Fetch and rank pages from pending URLs, adding relevant content to knowledge base.

    Flow:
    1. Take up to MAX_PAGES_PER_ITER from pending URLs
    2. Fetch and parse each page (async)
    3. Truncate content for reranker token limits
    4. Rerank all fetched chunks against the query
    5. Filter by RELEVANCE_THRESHOLD (keep MIN_CHUNKS_TO_KEEP minimum)
    6. Filter by MIN_SCORE_FOR_KB if KB already has content
    7. Deduplicate against existing KB entries
    8. Discover new links for next iteration
    """
    pending = state["pending_urls"][:MAX_PAGES_PER_ITER]
    visited = state["visited_urls"] | set(pending)

    print(f"\n{'─' * 60}")
    print(f"[RETRIEVE] ▶ Iteration {state['iteration'] + 1}/{MAX_ITERATIONS}")
    print(f"[RETRIEVE] Pending URLs: {pending}")
    print(f"[RETRIEVE] Already visited: {len(state['visited_urls'])} URLs")

    new_chunks: list[KBEntry] = []
    all_new_links: list[str] = []
    fetch_metadata: list[FetchMeta] = []

    tasks = [fetch_and_parse(url) for url in pending]
    results = await asyncio.gather(*tasks)

    for url, (text, links, meta) in zip(pending, results):
        fetch_metadata.append(meta)
        if text:
            new_chunks.append(
                {
                    "url": meta.get("final_url") or url,
                    "content": text,
                    "score": 0.0,
                    "raw_score": 0.0,
                    "original_chars": len(text),
                    "truncated_chars": len(text),
                }
            )
            all_new_links.extend([l for l in links if l not in visited])
        elif meta.get("error"):
            print(f"[RETRIEVE] ⚠️  Skipping {url}: fetch failed")
        else:
            print(
                f"[RETRIEVE] ⚠️  Skipping {url}: empty content ({meta['text_len']} chars)"
            )

    print(
        f"[RETRIEVE] Fetched {len(new_chunks)} pages with content, "
        f"discovered {len(all_new_links)} new links"
    )

    # Truncate for reranker, then score
    new_chunks = prepare_docs_for_rerank(new_chunks)
    ranked_chunks = rerank_chunks(state["query"], new_chunks)

    # Filter by relevance threshold, keeping minimum
    above_threshold = [c for c in ranked_chunks if c["score"] >= RELEVANCE_THRESHOLD]
    if len(above_threshold) < MIN_CHUNKS_TO_KEEP and ranked_chunks:
        relevant = ranked_chunks[: max(MIN_CHUNKS_TO_KEEP, len(above_threshold))]
        print(
            f"[RETRIEVE] ⚡ Only {len(above_threshold)} chunks above threshold "
            f"({RELEVANCE_THRESHOLD}), keeping top {len(relevant)} by score"
        )
    else:
        relevant = above_threshold
        print(
            f"[RETRIEVE] ✓ {len(relevant)} chunks above threshold "
            f"({RELEVANCE_THRESHOLD})"
        )

    # Additional filtering when KB already has content
    if state["knowledge_base"]:
        before_filter = len(relevant)
        relevant = [c for c in relevant if c["score"] >= MIN_SCORE_FOR_KB]
        dropped = before_filter - len(relevant)
        if dropped > 0:
            print(
                f"[RETRIEVE] 🗑️  Dropped {dropped} low-score chunks "
                f"(below {MIN_SCORE_FOR_KB}) since KB already has content"
            )

    # Deduplicate by URL
    existing_urls = {k["url"] for k in state["knowledge_base"]}
    new_relevant = [c for c in relevant if c["url"] not in existing_urls]
    merged_kb = state["knowledge_base"] + new_relevant

    # Prepare next iteration's pending URLs
    deduped_pending = list(set(all_new_links) - visited)[:10]

    print(
        f"[RETRIEVE] Summary: fetched={len(new_chunks)}, relevant={len(relevant)}, "
        f"new_to_kb={len(new_relevant)}, kb_total={len(merged_kb)}, "
        f"next_pending={len(deduped_pending)}"
    )

    if not deduped_pending and state["iteration"] == 0:
        print(
            "[RETRIEVE] ⚠️  WARNING: No pending URLs after first iteration! "
            "Check FOLLOW_CROSS_DOMAIN_LINKS or root URL validity."
        )

    return {
        "knowledge_base": merged_kb,
        "visited_urls": visited,
        "pending_urls": deduped_pending,
        "iteration": state["iteration"] + 1,
    }


async def evaluate_node(state: WebSwarmState) -> EvaluateNodeOutput:
    """Evaluate if the knowledge base contains sufficient information.

    Uses the LLM to assess whether retrieved content answers the query,
    considering remaining iterations and pending URLs.
    """
    print(
        f"\n[EVALUATE] Assessing KB size={len(state['knowledge_base'])} "
        f"at iteration {state['iteration']}"
    )

    kb_summary = "\n---\n".join(
        f"[{c['url']}] (score:{c['score']:.4f}, raw:{c.get('raw_score', 'N/A')})\n"
        f"{c['content'][:500]}"
        for c in state["knowledge_base"][:10]
    )

    has_pending = bool(state.get("pending_urls"))
    print(
        f"[EVALUATE] Pending URLs available: {has_pending} "
        f"({len(state.get('pending_urls', []))} remaining)"
    )

    prompt = f"""You are a RAG evaluation agent. Determine if the retrieved context sufficiently answers the query.

QUERY: {state["query"]}
ROOT URL: {state["root_url"]}
ITERATION: {state["iteration"]}/{MAX_ITERATIONS}
PENDING_URLS_AVAILABLE: {has_pending}

RETRIEVED CONTEXT:
{kb_summary if kb_summary else "(No relevant context retrieved yet)"}

Respond with ONLY valid JSON:
{{"evaluation": "sufficient|insufficient|irrelevant", "reasoning": "brief explanation"}}

Rules:
- "sufficient": Context contains SPECIFIC FACTS, DETAILS, or ENUMERATIONS that directly answer the query. A link or mention that information exists elsewhere is NOT sufficient.
- "insufficient": Context touches on the topic but lacks specific details, OR only provides links/references to where the answer might be found AND pending_urls remain.
- "irrelevant": Retrieved content is completely off-topic OR max iterations reached without finding relevant content.

CRITICAL: If the query asks "what are the X options/features/types/steps", the context must LIST or DESCRIBE those specific items. A page titled "Overview" or "Introduction" that says "see documentation for details" does NOT satisfy a question asking for those specific details. The context must contain the actual answer, not just a pointer to it."""

    content = llm_completion(
        prompt=prompt,
        temperature=0,
        response_format={"type": "json_object"},
        label="EVALUATE",
    )

    result: EvalResult = json.loads(content)
    evaluation = result.get("evaluation", "insufficient")
    reasoning = result.get("reasoning", "unknown")

    print(f"[EVALUATE] Decision: {evaluation} | Reason: {reasoning}")
    return {"evaluation": evaluation}


async def synthesize_node(state: WebSwarmState) -> SynthesizeNodeOutput:
    """Generate the final answer from the top-ranked knowledge base entries.

    Uses the LLM to synthesize a comprehensive answer citing sources.
    """
    sorted_kb = sorted(state["knowledge_base"], key=lambda x: x["score"], reverse=True)[
        :8
    ]

    print(f"\n[SYNTHESIZE] Generating answer from {len(sorted_kb)} KB entries")
    for i, c in enumerate(sorted_kb):
        print(f"  Source [{i}]: {c['url']} (score={c['score']:.4f})")

    context = "\n\n===SOURCE===".join(
        f"URL: {c['url']}\nRelevance: {c['score']:.4f}\n{c['content']}"
        for c in sorted_kb
    )

    prompt = f"""Using ONLY the provided context, answer the query comprehensively.
Cite sources as [URL]. If context is insufficient, say so explicitly.
Do NOT use any knowledge outside the provided context.

QUERY: {state["query"]}

CONTEXT:
{context if context else "(No relevant context was retrieved during the search.)"}"""

    content = llm_completion(
        prompt=prompt,
        temperature=0.1,
        label="SYNTHESIZE",
    )

    return {"final_answer": content}


def should_continue(state: WebSwarmState) -> Literal["retrieve", "synthesize", END]:
    """Decide the next graph step based on evaluation and state.

    Routing logic:
    - Sufficient/irrelevant → synthesize
    - Max iterations reached → synthesize
    - No pending URLs → synthesize
    - Otherwise → retrieve (continue crawling)
    """
    decision: Literal["retrieve", "synthesize"]
    reason: str

    if state["evaluation"] == "sufficient":
        decision = "synthesize"
        reason = "evaluation=sufficient"
    elif state["evaluation"] == "irrelevant":
        decision = "synthesize"
        reason = "evaluation=irrelevant"
    elif state["iteration"] >= MAX_ITERATIONS:
        decision = "synthesize"
        reason = f"max_iterations={MAX_ITERATIONS} reached"
    elif not state["pending_urls"]:
        decision = "synthesize"
        reason = "no_pending_urls"
    else:
        decision = "retrieve"
        reason = (
            f"evaluation={state['evaluation']}, pending={len(state['pending_urls'])}"
        )

    print(f"\n[ROUTE] → {decision.upper()} ({reason})")
    print(f"{'─' * 60}")
    return decision


def build_webswarm_graph() -> CompiledStateGraph[WebSwarmState]:
    """Build the LangGraph workflow for the WebSwarm RAG pipeline.

    Graph structure:
        START → retrieve → evaluate → [conditional] → synthesize → END
                                     ├─ continue → retrieve (loop)
                                     └─ stop → synthesize
    """
    workflow = StateGraph(WebSwarmState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("synthesize", synthesize_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "evaluate")
    workflow.add_conditional_edges("evaluate", should_continue)
    workflow.add_edge("synthesize", END)

    return workflow.compile()


async def run_webswarm(query: str, root_url: str) -> SwarmResult:
    """Execute the WebSwarm RAG pipeline end-to-end.

    Args:
        query: Research question to answer
        root_url: Starting URL for crawling

    Returns:
        SwarmResult with answer, sources, trace data, and configuration
    """
    app = build_webswarm_graph()

    config_snapshot = {
        "max_iterations": MAX_ITERATIONS,
        "max_pages_per_iter": MAX_PAGES_PER_ITER,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "min_chunks_to_keep": MIN_CHUNKS_TO_KEEP,
        "min_score_for_kb": MIN_SCORE_FOR_KB,
        "apply_sigmoid_normalization": APPLY_SIGMOID_NORMALIZATION,
        "follow_cross_domain_links": FOLLOW_CROSS_DOMAIN_LINKS,
        "rerank_max_tokens": RERANK_MAX_TOKENS,
        "llm_model": LLM_MODEL,
        "llm_base_url": LLM_BASE_URL,
        "rerank_model": RERANK_MODEL,
        "rerank_base_url": RERANK_BASE_URL,
    }

    print(f"\n{'═' * 60}")
    print(f'[INIT] Query: "{query}"')
    print(f"[INIT] Root URL: {root_url}")
    print(
        f"[INIT] Config: max_iter={MAX_ITERATIONS}, pages/iter={MAX_PAGES_PER_ITER}, "
        f"threshold={RELEVANCE_THRESHOLD}, min_keep={MIN_CHUNKS_TO_KEEP}, "
        f"min_score_kb={MIN_SCORE_FOR_KB}, sigmoid={APPLY_SIGMOID_NORMALIZATION}, "
        f"cross_domain={FOLLOW_CROSS_DOMAIN_LINKS}, "
        f"rerank_max_tokens={RERANK_MAX_TOKENS}"
    )
    print(f"[INIT] LLM: {LLM_MODEL} @ {LLM_BASE_URL}")
    print(f"[INIT] Reranker: {RERANK_MODEL} @ {RERANK_BASE_URL}")
    print(f"{'═' * 60}")

    initial_state: WebSwarmState = {
        "query": query,
        "root_url": root_url,
        "messages": [],
        "visited_urls": set(),
        "pending_urls": [root_url],
        "knowledge_base": [],
        "iteration": 0,
        "evaluation": "insufficient",
        "final_answer": None,
    }

    invoke_config = {"recursion_limit": MAX_ITERATIONS * 3 + 5}

    trace: dict = {
        "retrieve_steps": [],
        "evaluate_steps": [],
        "synthesize_step": None,
        "route_decisions": [],
    }

    # === Node tracing wrappers ===
    original_retrieve = retrieve_node
    original_evaluate = evaluate_node
    original_synthesize = synthesize_node
    original_route = should_continue

    async def traced_retrieve(state: WebSwarmState) -> RetrieveNodeOutput:
        result = await original_retrieve(state)
        step: RetrieveStep = {
            "iteration": result["iteration"],
            "pending_urls": list(state["pending_urls"][:MAX_PAGES_PER_ITER]),
            "visited_count": len(state["visited_urls"]),
            "fetched_pages": len(result["knowledge_base"])
            - len(state["knowledge_base"]),
            "discovered_links": len(result["pending_urls"]),
            "reranked_chunks": [
                {
                    "url": c["url"],
                    "score": round(c["score"], 4),
                    "raw_score": round(c.get("raw_score", 0.0), 4),
                    "truncated_chars": c.get("truncated_chars", 0),
                }
                for c in result["knowledge_base"]
                if c["url"] not in {k["url"] for k in state["knowledge_base"]}
            ],
            "chunks_above_threshold": sum(
                1
                for c in result["knowledge_base"]
                if c["score"] >= RELEVANCE_THRESHOLD
                and c["url"] not in {k["url"] for k in state["knowledge_base"]}
            ),
            "chunks_added_to_kb": len(result["knowledge_base"])
            - len(state["knowledge_base"]),
            "kb_total_after": len(result["knowledge_base"]),
            "next_pending_count": len(result["pending_urls"]),
        }
        trace["retrieve_steps"].append(step)
        return result

    async def traced_evaluate(state: WebSwarmState) -> EvaluateNodeOutput:
        result = await original_evaluate(state)
        step: EvaluateStep = {
            "iteration": state["iteration"],
            "kb_size": len(state["knowledge_base"]),
            "pending_available": bool(state.get("pending_urls")),
            "pending_count": len(state.get("pending_urls", [])),
            "evaluation": result["evaluation"],
            "reasoning": "",
        }
        trace["evaluate_steps"].append(step)
        return result

    async def traced_synthesize(state: WebSwarmState) -> SynthesizeNodeOutput:
        result = await original_synthesize(state)
        sorted_kb = sorted(
            state["knowledge_base"], key=lambda x: x["score"], reverse=True
        )[:8]
        step: SynthesizeStep = {
            "kb_entries_used": len(sorted_kb),
            "source_urls": [c["url"] for c in sorted_kb],
            "source_scores": [round(c["score"], 4) for c in sorted_kb],
            "answer": result["final_answer"],
        }
        trace["synthesize_step"] = step
        return result

    def traced_route(state: WebSwarmState) -> Literal["retrieve", "synthesize", END]:
        decision = original_route(state)

        if state["evaluation"] == "sufficient":
            reason = "evaluation=sufficient"
        elif state["evaluation"] == "irrelevant":
            reason = "evaluation=irrelevant"
        elif state["iteration"] >= MAX_ITERATIONS:
            reason = f"max_iterations={MAX_ITERATIONS} reached"
        elif not state["pending_urls"]:
            reason = "no_pending_urls"
        else:
            reason = (
                f"evaluation={state['evaluation']}, "
                f"pending={len(state['pending_urls'])}"
            )

        rd: RouteDecision = {
            "iteration": state["iteration"],
            "decision": decision if decision != END else "synthesize",
            "reason": reason,
        }
        trace["route_decisions"].append(rd)
        return decision

    # Monkey-patch for tracing
    import jet.agents.llama_cpp.rag_web_crawler_agent as self_module

    self_module.retrieve_node = traced_retrieve
    self_module.evaluate_node = traced_evaluate
    self_module.synthesize_node = traced_synthesize
    self_module.should_continue = traced_route

    try:
        traced_app = build_webswarm_graph()
        final_state = await traced_app.ainvoke(initial_state, config=invoke_config)
    finally:
        self_module.retrieve_node = original_retrieve
        self_module.evaluate_node = original_evaluate
        self_module.synthesize_node = original_synthesize
        self_module.should_continue = original_route

    print(f"\n{'═' * 60}")
    print(f"[DONE] Iterations: {final_state['iteration']}")
    print(f"[DONE] Pages visited: {len(final_state['visited_urls'])}")
    print(f"[DONE] KB entries: {len(final_state['knowledge_base'])}")
    print(f"[DONE] Sources: {[k['url'] for k in final_state['knowledge_base']]}")
    print(f"{'═' * 60}")

    return {
        "answer": final_state["final_answer"],
        "sources": [k["url"] for k in final_state["knowledge_base"]],
        "iterations": final_state["iteration"],
        "pages_visited": len(final_state["visited_urls"]),
        "knowledge_base": final_state["knowledge_base"],
        "config": config_snapshot,
        "retrieve_steps": trace["retrieve_steps"],
        "evaluate_steps": trace["evaluate_steps"],
        "synthesize_step": trace["synthesize_step"],
        "route_decisions": trace["route_decisions"],
    }


if __name__ == "__main__":
    result = asyncio.run(
        run_webswarm(
            query="What are the deployment options for LangGraph Platform?",
            root_url="https://langchain-ai.github.io/langgraph/",
        )
    )
    print(f"\nANSWER ({result['iterations']} iters, {result['pages_visited']} pages):")
    print(result["answer"])
    print(f"\nSOURCES: {result['sources']}")
