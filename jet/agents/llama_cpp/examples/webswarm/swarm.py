import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import asyncio
import hashlib
import json
import logging
import time
from typing import Any, TypedDict

import chromadb
import trafilatura
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from config import (
    BUDGETS,
    CACHE_DB,
    DOC_CHAR_LIMIT,
    GRAMMAR_DIR,
    MAX_DEPTH,
    MAX_ITERATIONS,
    MAX_TOTAL_TOKENS,
    MAX_WALL_SECONDS,
    RERANK_TOP_K,
    SEARXNG_CATEGORIES,
    SEARXNG_ENGINES,
    SEARXNG_MAX_RESULTS,
    SEARXNG_MIN_SCORE,
    SEARXNG_QUERY_URL,
    SEARXNG_USE_CACHE,
    SEMANTIC_DEDUP_THRESHOLD,
    VECTOR_DB_PATH,
)
from jet.adapters.llama_cpp.chunking_utils import chunk_texts_with_data, truncate_texts
from jet.adapters.llama_cpp.config import EMBED_LG_MODEL, LLM_MODEL
from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.factory import get_llm_client
from jet.adapters.llama_cpp.rerank_utils import rerank
from jet.adapters.llama_cpp.token_utils import count_tokens
from jet.search.searxng import search_searxng
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from playwright.async_api import async_playwright

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("webswarm")


DEFAULT_LLM_MODEL = LLM_MODEL
DEFAULT_EMBED_MODEL = EMBED_LG_MODEL


class JetEmbeddingFunction(EmbeddingFunction[Documents]):
    """ChromaDB-compatible wrapper for jet.adapters.llama_cpp.embed."""

    def __call__(self, input: Documents) -> Embeddings:
        return embed(
            input, model=DEFAULT_EMBED_MODEL, return_format="list", show_progress=True
        )

    def name(self) -> str:
        return "jet_llama_cpp_embed"


class LocalLLMClient:
    """Budget-aware wrapper using jet.adapters.llama_cpp.factory."""

    def __init__(self):
        self._client = get_llm_client()
        self.tokens_used = 0
        self._grammars = {}

    def _load_grammar(self, name: str) -> str:
        if name not in self._grammars:
            path = os.path.join(GRAMMAR_DIR, f"{name}.gbnf")
            if not os.path.isfile(path):
                available = (
                    os.listdir(GRAMMAR_DIR) if os.path.isdir(GRAMMAR_DIR) else []
                )
                raise FileNotFoundError(
                    f"Grammar file not found: {path}\n"
                    f"GRAMMAR_DIR={GRAMMAR_DIR}\n"
                    f"Available files: {available}"
                )
            self._grammars[name] = open(path).read()
            logger.debug(f"Loaded grammar '{name}' from {path}")
        return self._grammars[name]

    async def chat(
        self, messages: list[dict], grammar: str | None = None, max_tokens: int = 1024
    ) -> dict | str:
        kwargs: dict[str, Any] = {
            "model": LLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "top_p": 0.95,
            "presence_penalty": 1.5,
            "seed": 42,
            "stream": True,
        }
        if grammar:
            grammar_content = self._load_grammar(grammar)
            logger.debug(
                f"Grammar '{grammar}' payload: length={len(grammar_content)}, "
                f"rule_count={grammar_content.count('::=')}, "
                f"enable_thinking=False"
            )
            kwargs["extra_body"] = {
                "grammar": grammar_content,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        else:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        loop = asyncio.get_running_loop()
        stream = await loop.run_in_executor(
            None, lambda: self._client.chat.completions.create(**kwargs)
        )
        content_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        for chunk in stream:
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                content_parts.append(delta.content)
                print(delta.content, end="", flush=True)
        print()
        self.tokens_used += prompt_tokens + completion_tokens
        content = "".join(content_parts)
        if grammar and not content.strip():
            logger.error(
                f"EMPTY response with grammar '{grammar}'. "
                f"Prompt tokens: {prompt_tokens or 'unknown'}. "
                f"Verify enable_thinking=False is reaching the server."
            )
            return {"error": "EMPTY_RESPONSE", "raw": ""}
        if grammar:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Grammar output parse failed: {content[:200]}")
                return {"error": "PARSE_FAIL", "raw": content}
        return content


class LocalRetriever:
    """Uses jet.adapters.llama_cpp embed/rerank utils + ChromaDB."""

    def __init__(self):
        self.chroma = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        try:
            self.chroma.delete_collection("swarm_findings")
            logger.info("Deleted existing 'swarm_findings' collection")
        except ValueError:
            pass
        self.collection = self.chroma.get_or_create_collection(
            "swarm_findings",
            metadata={"hnsw:space": "cosine"},
            embedding_function=JetEmbeddingFunction(),
        )
        logger.info(
            "ChromaDB collection initialized with JetEmbeddingFunction (custom llama.cpp embeddings)"
        )

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Async wrapper around jet embed() with batch dedup and VRAM safety."""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: embed(
                texts,
                model=DEFAULT_EMBED_MODEL,
                return_format="list",
                show_progress=True,
            ),
        )
        return result

    async def rerank_docs(
        self, query: str, docs: list[dict], top_k: int = RERANK_TOP_K
    ) -> list[dict]:
        """Async wrapper around jet rerank() using correct /rerank endpoint."""
        if not docs:
            return []
        doc_texts = [d.get("text", d.get("content", ""))[:DOC_CHAR_LIMIT] for d in docs]
        loop = asyncio.get_running_loop()
        rerank_results = await loop.run_in_executor(
            None, lambda: rerank(query, doc_texts, top_n=min(top_k, len(docs)))
        )
        return [docs[r["index"]] for r in rerank_results]

    async def recall_with_chunks(
        self,
        query: str,
        top_k: int = 3,
        branch_filter: str = None,
        merge_chunks: bool = True,
    ) -> list[dict]:
        """Enhanced recall that handles chunked storage.

        Args:
            merge_chunks: If True, merge chunks from same finding back together.
                        If False, return individual chunks.
        """
        where = {"branch": branch_filter} if branch_filter else None
        logger.debug(f"Recall query with chunking support")

        # Get more results since we might have multiple chunks per finding
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k * 3,  # Get extra to account for chunks
            where=where,
        )

        if not results["documents"][0]:
            return []

        if not merge_chunks:
            return [
                {
                    "content": d,
                    "url": m.get("url", ""),
                    "score": s,
                    "chunk_index": m.get("chunk_index"),
                    "subtask_id": m.get("subtask_id"),
                }
                for d, m, s in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ]

        # Merge chunks from the same finding
        findings_map: dict[str, dict] = {}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            subtask_id = meta.get("subtask_id", "unknown")

            if subtask_id not in findings_map:
                findings_map[subtask_id] = {
                    "content": "",
                    "url": meta.get("url", ""),
                    "score": dist,
                    "chunks": [],
                    "subtask_id": subtask_id,
                }

            findings_map[subtask_id]["chunks"].append(
                {
                    "content": doc,
                    "score": dist,
                    "chunk_index": meta.get("chunk_index", 0),
                }
            )
            # Keep best score
            findings_map[subtask_id]["score"] = min(
                findings_map[subtask_id]["score"], dist
            )

        # Sort chunks and merge content
        merged_findings = []
        for finding in sorted(findings_map.values(), key=lambda x: x["score"])[:top_k]:
            finding["chunks"].sort(key=lambda x: x["chunk_index"])
            finding["content"] = "\n\n".join(c["content"] for c in finding["chunks"])
            merged_findings.append(
                {
                    "content": finding["content"],
                    "url": finding["url"],
                    "score": finding["score"],
                    "subtask_id": finding["subtask_id"],
                }
            )

        logger.info(
            f"Recalled {len(merged_findings)} merged findings from "
            f"{sum(len(f['chunks']) for f in findings_map.values())} chunks"
        )

        return merged_findings


async def store_finding(self, finding: dict, use_chunking: bool = True):
    """Store finding with smart chunking for better retrieval.

    Args:
        finding: Finding dict with content, subtask_id, etc.
        use_chunking: If True, split content into semantic chunks.
                     If False, fall back to original behavior.
    """
    content = finding["content"]
    if not content:
        logger.warning(f"Empty content for subtask '{finding['subtask_id']}', skipping")
        return

    subtask_id = finding["subtask_id"]
    metadata_base = {
        "url": finding.get("url", ""),
        "branch": finding.get("branch_id", ""),
        "confidence": finding.get("confidence", "NONE"),
        "subtask_id": subtask_id,
    }

    if use_chunking:
        # Chunk content with sentence awareness and overlap
        chunks = chunk_texts_with_data(
            content,
            chunk_size=256,  # Adjust based on your embedding model's context
            chunk_overlap=50,  # Overlap for context continuity
            model=LLM_MODEL,
            strict_sentences=True,
            min_chunk_size=64,
            show_progress=False,
        )

        if not chunks:
            logger.warning(f"No chunks generated for subtask '{subtask_id}'")
            return

        logger.info(
            f"Chunked finding '{subtask_id}' into {len(chunks)} pieces "
            f"(avg {sum(c['num_tokens'] for c in chunks) // len(chunks)} tokens/chunk)"
        )

        # Store each chunk with its metadata
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{subtask_id}_chunk_{idx}"
            try:
                emb_list = await self.embed_texts([chunk["content"]])
                if not emb_list:
                    logger.error(f"Embedding failed for chunk {idx} of '{subtask_id}'")
                    continue

                self.collection.upsert(
                    ids=[chunk_id],
                    embeddings=[emb_list[0]],
                    documents=[chunk["content"]],
                    metadatas=[
                        {
                            **metadata_base,
                            "chunk_index": idx,
                            "total_chunks": len(chunks),
                            "token_count": chunk["num_tokens"],
                        }
                    ],
                )
                logger.debug(
                    f"Stored chunk {idx}/{len(chunks)} of '{subtask_id}' "
                    f"({chunk['num_tokens']} tokens)"
                )
            except Exception as e:
                logger.error(f"Failed to store chunk {idx} of '{subtask_id}': {e}")
    else:
        # Original behavior with token-aware truncation
        token_count = count_tokens(content, model=DEFAULT_EMBED_MODEL)
        max_embed_tokens = 500

        if token_count > max_embed_tokens:
            logger.warning(
                f"Content too long ({token_count} tokens). Using smart truncation."
            )
            content = truncate_texts(
                content,
                model=LLM_MODEL,
                max_tokens=max_embed_tokens,
                strict_sentences=True,
                show_progress=False,
            )
            token_count = count_tokens(content, model=DEFAULT_EMBED_MODEL)
            logger.info(f"Truncated to {token_count} tokens")

        try:
            emb_list = await self.embed_texts([content])
            if not emb_list:
                logger.error(f"Embedding failed for '{subtask_id}'")
                return

            self.collection.upsert(
                ids=[subtask_id],
                embeddings=[emb_list[0]],
                documents=[content],
                metadatas=[{**metadata_base, "token_count": token_count}],
            )
            logger.debug(
                f"Stored finding '{subtask_id}' ({len(content)} chars, {token_count} tokens)"
            )
        except Exception as e:
            logger.error(f"Failed to store finding '{subtask_id}': {e}")

    async def recall(
        self, query: str, top_k: int = 3, branch_filter: str = None
    ) -> list[dict]:
        where = {"branch": branch_filter} if branch_filter else None
        logger.debug(f"Recall query using registered custom embedding function")
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


async def web_search(query: str) -> list[str]:
    """Async wrapper around jet.search.searxng.search_searxng."""
    loop = asyncio.get_running_loop()
    try:
        results = await loop.run_in_executor(
            None,
            lambda: search_searxng(
                query=query,
                query_url=SEARXNG_QUERY_URL,
                count=SEARXNG_MAX_RESULTS,
                min_score=SEARXNG_MIN_SCORE,
                engines=SEARXNG_ENGINES,
                categories=SEARXNG_CATEGORIES,
                use_cache=SEARXNG_USE_CACHE,
            ),
        )
        urls = [r["url"] for r in results if r.get("url")]
        logger.info(f"SearXNG returned {len(urls)} URLs for: {query[:80]}")
        return urls
    except Exception as e:
        logger.error(f"SearXNG search failed for '{query[:80]}': {e}")
        return []


llm = LocalLLMClient()
retriever = LocalRetriever()
dedup = DedupCache()


async def _safe_llm_call(
    messages: list[dict], role: str, grammar: str = None
) -> dict | str:
    """Context degradation cascade using smart truncation."""
    budget = BUDGETS[role]
    total = count_tokens(messages, model=DEFAULT_LLM_MODEL)
    limit = sum(budget.values())

    if total <= limit:
        return await llm.chat(messages, grammar=grammar, max_tokens=budget["output"])

    logger.warning(f"[{role}] Context overflow ({total}>{limit}). Trimming docs.")

    # Find the longest user message (usually contains the documents)
    user_msgs = [(i, m) for i, m in enumerate(messages) if m["role"] == "user"]
    user_msgs.sort(key=lambda x: len(x[1].get("content", "")), reverse=True)

    if user_msgs:
        idx, msg = user_msgs[0]
        content = msg["content"]

        # Calculate how many tokens we need to remove
        excess = total - limit
        target_tokens = count_tokens(content, model=DEFAULT_LLM_MODEL) - excess

        # Use smart truncation that preserves sentence boundaries
        if target_tokens > 0:
            trimmed = truncate_texts(
                content,
                model=LLM_MODEL,
                max_tokens=target_tokens,
                strict_sentences=True,
                show_progress=False,
            )
            messages[idx] = {
                "role": "user",
                "content": trimmed
                + "\n\n[Note: Content truncated to fit context window]",
            }
            logger.info(
                f"[{role}] Truncated content from {len(content)} to {len(trimmed)} chars "
                f"({count_tokens(trimmed, model=DEFAULT_LLM_MODEL)} tokens)"
            )

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
            "content": f"Query: {state['query']}\nPrior findings:\n{history_summary}",
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

    # Use enhanced recall with chunk merging
    recalled = await retriever.recall_with_chunks(
        task["question"],
        top_k=1,
        merge_chunks=True,  # Merge chunks from same finding
    )

    if recalled and recalled[0].get("score", 1.0) < (1 - SEMANTIC_DEDUP_THRESHOLD):
        logger.info(
            f"Dedup hit for '{task['question'][:60]}' "
            f"(score: {recalled[0]['score']:.3f})"
        )
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
    urls = await web_search(task["question"])
    candidates = [{"text": "", "url": u} for u in urls]
    ranked = await retriever.rerank_docs(task["question"], candidates)
    finding_content = ""
    finding_url = ""
    if ranked:
        page = await extract_page(ranked[0]["url"])
        finding_content = page["text"]
        finding_url = page["url"]
    dedup.mark_seen(task["question"], finding_url)
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

    # Use recall_with_chunks for semantic retrieval instead of direct finding access
    recalled_findings = await retriever.recall_with_chunks(
        state["query"], top_k=5, merge_chunks=True
    )

    # Fall back to reranking if recall returns nothing (e.g., empty collection)
    if recalled_findings:
        # Rerank the recalled findings for better ordering
        top_findings = await retriever.rerank_docs(
            state["query"], recalled_findings, top_k=5
        )
        logger.info(
            f"Synthesizer using {len(top_findings)} recalled & reranked findings"
        )
    else:
        # Fallback: use findings directly from state
        top_findings = await retriever.rerank_docs(
            state["query"], state.get("findings", []), top_k=5
        )
        logger.info(
            f"Synthesizer using {len(top_findings)} findings from state (no recall hits)"
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
            "content": f"Original query: {state['query']}\n"
            f"Global index:\n{global_index}\nDetailed findings:\n{detailed}",
        },
    ]
    answer = await _safe_llm_call(messages, "synthesizer")
    return {"final_answer": answer if isinstance(answer, str) else str(answer)}


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
        return "plan"
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
