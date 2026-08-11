import asyncio
import hashlib
import json
import logging
import os
import time
from typing import TypedDict

import chromadb
import httpx
import trafilatura
from config import *
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from openai import AsyncOpenAI
from playwright.async_api import async_playwright

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("webswarm")

# =============================================================================
# INFRASTRUCTURE CLIENTS
# =============================================================================


class LocalLLMClient:
    """Budget-aware wrapper around local llama.cpp server."""

    def __init__(self):
        self.client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key="none")
        self.tokens_used = 0
        self._grammars = {}

    def _load_grammar(self, name: str) -> str:
        if name not in self._grammars:
            path = os.path.join(GRAMMAR_DIR, f"{name}.gbnf")
            self._grammars[name] = open(path).read()
        return self._grammars[name]

    async def chat(
        self, messages: list[dict], grammar: str = None, max_tokens: int = 512
    ) -> dict:
        kwargs = {
            "model": LLM_MODEL_NAME,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }
        if grammar:
            kwargs["extra_body"] = {"grammar": self._load_grammar(grammar)}

        resp = await self.client.chat.completions.create(**kwargs)
        usage = resp.usage
        self.tokens_used += usage.prompt_tokens + usage.completion_tokens
        content = resp.choices[0].message.content

        # Safety parse for grammar-constrained outputs
        if grammar:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Grammar output parse failed: {content[:200]}")
                return {"error": "PARSE_FAIL", "raw": content}
        return content


class LocalRetriever:
    """Embedder + Reranker + Vector Store integration."""

    def __init__(self):
        self.chroma = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self.collection = self.chroma.get_or_create_collection(
            "swarm_findings", metadata={"hnsw:space": "cosine"}
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(EMBEDDER_URL, json={"texts": texts})
            return r.json()["embeddings"]

    async def rerank(
        self, query: str, docs: list[dict], top_k: int = RERANK_TOP_K
    ) -> list[dict]:
        if not docs:
            return []
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                RERANKER_URL,
                json={
                    "query": query,
                    "documents": [
                        d.get("text", d.get("content", ""))[:DOC_CHAR_LIMIT]
                        for d in docs
                    ],
                    "top_k": min(top_k, len(docs)),
                },
            )
            results = r.json()["results"]
            return [
                docs[i] for i, _ in sorted(results, key=lambda x: x[1], reverse=True)
            ]

    async def store_finding(self, finding: dict):
        emb = (await self.embed([finding["content"][:DOC_CHAR_LIMIT]]))[0]
        self.collection.upsert(
            ids=[finding["subtask_id"]],
            embeddings=[emb],
            documents=[finding["content"][:DOC_CHAR_LIMIT]],
            metadatas=[
                {"url": finding.get("url", ""), "branch": finding.get("branch_id", "")}
            ],
        )

    async def recall(
        self, query: str, top_k: int = 3, branch_filter: str = None
    ) -> list[dict]:
        where = {"branch": branch_filter} if branch_filter else None
        results = self.collection.query(
            query_texts=[query], n_results=top_k, where=where
        )
        if not results["documents"][0]:
            return []
        return [
            {"content": d, "url": m.get("url", ""), "score": s}
            for d, m, s in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]


class BrowserManager:
    _ctx = None

    @classmethod
    async def get_context(cls):
        if cls._ctx is None:
            pw = await async_playwright().start()
            browser = await pw.chromium.launch(headless=True)
            cls._ctx = await browser.new_context(
                user_agent="Mozilla/5.0 (ResearchBot/1.0)",
                viewport={"width": 1280, "height": 800},
            )
        return cls._ctx


async def extract_page(url: str) -> dict:
    ctx = await BrowserManager.get_context()
    page = await ctx.new_page()
    try:
        await page.goto(url, timeout=15000, wait_until="domcontentloaded")
        html = await page.content()
        text = trafilatura.extract(html, include_comments=False) or ""
        return {"url": url, "text": text[:DOC_CHAR_LIMIT]}
    except Exception as e:
        logger.warning(f"Browser fail {url}: {e}")
        return {"url": url, "text": "", "error": str(e)}
    finally:
        await page.close()


# =============================================================================
# STATE & DEDUP
# =============================================================================


class SwarmState(TypedDict):
    query: str
    subtasks: list[dict]
    findings: list[dict]
    iteration: int
    tokens_used: int
    start_time: float
    final_answer: str | None


class DedupCache:
    def __init__(self):
        import sqlite3

        self.conn = sqlite3.connect(CACHE_DB)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS seen (hash TEXT PRIMARY KEY, ts REAL)"
        )

    def is_seen(self, query: str, url: str = "") -> bool:
        h = hashlib.sha256(f"{query}|{url}".encode()).hexdigest()
        return (
            self.conn.execute("SELECT 1 FROM seen WHERE hash=?", (h,)).fetchone()
            is not None
        )

    def mark_seen(self, query: str, url: str = ""):
        h = hashlib.sha256(f"{query}|{url}".encode()).hexdigest()
        self.conn.execute("INSERT OR IGNORE INTO seen VALUES (?, ?)", (h, time.time()))
        self.conn.commit()


# =============================================================================
# GRAPH NODES
# =============================================================================

llm = LocalLLMClient()
retriever = LocalRetriever()
dedup = DedupCache()


def _count_tokens_approx(text: str) -> int:
    """Rough token estimate for budget checks. Replace with llama-tokenize for precision."""
    return len(text) // 4


async def _safe_llm_call(
    messages: list[dict], role: str, grammar: str = None
) -> dict | str:
    """Context degradation cascade."""
    budget = BUDGETS[role]
    total = sum(_count_tokens_approx(m.get("content", "")) for m in messages)
    limit = sum(budget.values())

    if total <= limit:
        return await llm.chat(messages, grammar=grammar, max_tokens=budget["output"])

    # Level 1: Trim longest user message (usually docs)
    logger.warning(f"[{role}] Context overflow ({total}>{limit}). Trimming docs.")
    user_msgs = [(i, m) for i, m in enumerate(messages) if m["role"] == "user"]
    user_msgs.sort(key=lambda x: len(x[1].get("content", "")), reverse=True)
    if user_msgs:
        idx, msg = user_msgs[0]
        trimmed = msg["content"][: int(len(msg["content"]) * limit / total)]
        messages[idx] = {"role": "user", "content": trimmed + "\n[TRUNCATED]"}

    return await llm.chat(messages, grammar=grammar, max_tokens=budget["output"])


async def planner_node(state: SwarmState) -> dict:
    existing = state.get("findings", [])
    history_summary = ""
    if existing:
        lines = [
            f"- [{f['subtask_id']}] {f.get('summary', f['content'][:80])}"
            for f in existing
        ]
        history_summary = "\n".join(lines)[: BUDGETS["planner"]["history"] * 4]

    messages = [
        {
            "role": "system",
            "content": "You decompose research queries into subtasks. "
            "If prior findings exist, identify GAPS only. Output JSON per grammar.",
        },
        {
            "role": "user",
            "content": f"Query: {state['query']}\n\nPrior findings:\n{history_summary}",
        },
    ]
    result = await _safe_llm_call(messages, "planner", grammar="planner")
    if isinstance(result, dict) and "error" in result:
        logger.error(f"Planner failed: {result}")
        return {"subtasks": state.get("subtasks", [])}

    new_tasks = result.get("subtasks", [])
    for t in new_tasks:
        t.setdefault("branch_id", f"branch_{state['iteration']}")
    return {
        "subtasks": state.get("subtasks", []) + new_tasks,
        "iteration": state.get("iteration", 0) + 1,
    }


async def searcher_node(state: SwarmState) -> dict:
    answered_ids = {f["subtask_id"] for f in state.get("findings", [])}
    task = next((t for t in state["subtasks"] if t["id"] not in answered_ids), None)
    if not task:
        return {}

    # Semantic dedup
    recalled = await retriever.recall(task["question"], top_k=1)
    if recalled and recalled[0].get("score", 1.0) < (1 - SEMANTIC_DEDUP_THRESHOLD):
        logger.info(f"Dedup hit for '{task['question'][:60]}'")
        return {
            "findings": state.get("findings", [])
            + [
                {
                    "subtask_id": task["id"],
                    "content": recalled[0]["content"],
                    "url": recalled[0].get("url", ""),
                    "confidence": "RECALLED",
                    "summary": recalled[0]["content"][:100],
                    "branch_id": task.get("branch_id"),
                }
            ]
        }

    # Web search + rerank
    # NOTE: Replace with your actual search API call here
    # For now, simulating candidate URLs from a search
    candidates = [{"text": "", "url": u} for u in await _mock_search(task["question"])]
    ranked = await retriever.rerank(task["question"], candidates)

    # Browse top result
    finding_content = ""
    finding_url = ""
    if ranked:
        page = await extract_page(ranked[0]["url"])
        finding_content = page["text"]
        finding_url = page["url"]
        dedup.mark_seen(task["question"], finding_url)

    # Confidence evaluation
    conf_messages = [
        {
            "role": "system",
            "content": "Evaluate if content answers the question. Output JSON per grammar.",
        },
        {
            "role": "user",
            "content": f"Question: {task['question']}\nContent: {finding_content[:3000]}",
        },
    ]
    conf = await _safe_llm_call(conf_messages, "searcher", grammar="confidence")
    verdict = conf.get("verdict", "NONE") if isinstance(conf, dict) else "NONE"

    # Compress for future children
    comp_messages = [
        {
            "role": "system",
            "content": "Compress findings for child agents. Output JSON per grammar.",
        },
        {
            "role": "user",
            "content": f"Compress for: '{task['question']}'\n{finding_content[:3000]}",
        },
    ]
    compressed = await _safe_llm_call(comp_messages, "compressor", grammar="compressor")
    summary = (
        compressed.get("summary", finding_content[:100])
        if isinstance(compressed, dict)
        else finding_content[:100]
    )

    new_finding = {
        "subtask_id": task["id"],
        "content": finding_content,
        "url": finding_url,
        "confidence": verdict,
        "summary": summary,
        "branch_id": task.get("branch_id"),
    }
    await retriever.store_finding(new_finding)
    return {"findings": state.get("findings", []) + [new_finding]}


async def synthesizer_node(state: SwarmState) -> dict:
    global_index = "\n".join(
        f"- [{f['subtask_id']}] ({f['confidence']}) {f.get('summary', '')}"
        for f in state.get("findings", [])
    )[: BUDGETS["synthesizer"]["global_index"] * 4]

    top_findings = await retriever.rerank(
        state["query"], state.get("findings", []), top_k=5
    )
    detailed = "\n---\n".join(f["content"][:2000] for f in top_findings)

    messages = [
        {
            "role": "system",
            "content": "Synthesize a comprehensive, cited answer from findings. "
            "Acknowledge gaps honestly.",
        },
        {
            "role": "user",
            "content": f"Original query: {state['query']}\n\n"
            f"Global index:\n{global_index}\n\nDetailed findings:\n{detailed}",
        },
    ]
    answer = await _safe_llm_call(messages, "synthesizer")
    return {"final_answer": answer if isinstance(answer, str) else str(answer)}


# =============================================================================
# ROUTING & GRAPH COMPILATION
# =============================================================================


def should_recurse(state: SwarmState) -> str:
    elapsed = time.time() - state.get("start_time", time.time())
    if (
        elapsed > MAX_WALL_SECONDS
        or llm.tokens_used > MAX_TOTAL_TOKENS
        or state.get("iteration", 0) >= MAX_ITERATIONS
    ):
        logger.warning(
            f"Budget exhausted. Elapsed={elapsed:.0f}s Tokens={llm.tokens_used} Iter={state.get('iteration')}"
        )
        return "synthesize"

    answered = {f["subtask_id"] for f in state.get("findings", [])}
    unanswered = [t for t in state.get("subtasks", []) if t["id"] not in answered]
    partial = [f for f in state.get("findings", []) if f.get("confidence") == "PARTIAL"]

    if unanswered:
        max_d = max((t.get("depth", 0) for t in unanswered), default=0)
        if max_d < MAX_DEPTH:
            return "search"
    if partial:
        return "plan"  # Recursive refinement
    if not unanswered:
        return "synthesize"
    return "synthesize"


async def run_swarm(query: str) -> str:
    graph = StateGraph(SwarmState)
    graph.add_node("plan", planner_node)
    graph.add_node("search", searcher_node)
    graph.add_node("synthesize", synthesizer_node)
    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", lambda _: "search", {"search": "search"})
    graph.add_conditional_edges(
        "search",
        should_recurse,
        {"search": "search", "plan": "plan", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)

    app = graph.compile(checkpointer=MemorySaver())
    initial_state = {
        "query": query,
        "subtasks": [],
        "findings": [],
        "iteration": 0,
        "tokens_used": 0,
        "start_time": time.time(),
        "final_answer": None,
    }
    result = await app.ainvoke(
        initial_state, config={"configurable": {"thread_id": query[:50]}}
    )
    return result.get("final_answer", "No answer generated.")


# Placeholder: Replace with real search API
async def _mock_search(query: str) -> list[str]:
    """Replace with SerpAPI, SearXNG, or DuckDuckGo integration."""
    logger.info(f"[MOCK SEARCH] {query}")
    return []  # Return list of URLs


# === ENTRY POINT ===

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run WebSwarm with a query.")
    parser.add_argument(
        "query",
        nargs="?",
        default="What are the supply chain risks for solid-state batteries in SE Asia?",
        help="The query to run WebSwarm on.",
    )
    args = parser.parse_args()

    answer = asyncio.run(run_swarm(args.query))
    print("\n" + "=" * 80 + "\n" + answer)
