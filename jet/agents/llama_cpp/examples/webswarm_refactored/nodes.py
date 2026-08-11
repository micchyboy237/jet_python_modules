import logging
import time

from .browser import extract_page
from .cache import DedupCache
from .config import BUDGETS, MAX_DEPTH, SEMANTIC_DEDUP_THRESHOLD
from .llm_client import LocalLLMClient, safe_llm_call
from .retriever import LocalRetriever
from .search import web_search
from .state import SwarmState

logger = logging.getLogger("webswarm")


async def planner_node(state: SwarmState, llm: LocalLLMClient) -> dict:
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

    result = await safe_llm_call(llm, messages, "planner", grammar="planner")
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


async def searcher_node(
    state: SwarmState,
    llm: LocalLLMClient,
    retriever: LocalRetriever,
    dedup: DedupCache,
) -> dict:
    answered_ids = {f["subtask_id"] for f in state.get("findings", [])}
    task = next((t for t in state["subtasks"] if t["id"] not in answered_ids), None)
    if not task:
        return {}

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
    conf = await safe_llm_call(llm, conf_messages, "searcher", grammar="confidence")
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
    compressed = await safe_llm_call(
        llm, comp_messages, "compressor", grammar="compressor"
    )
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


async def synthesizer_node(
    state: SwarmState, llm: LocalLLMClient, retriever: LocalRetriever
) -> dict:
    global_index = "\n".join(
        f"- [{f['subtask_id']}] ({f['confidence']}) {f.get('summary', '')}"
        for f in state.get("findings", [])
    )[: BUDGETS["synthesizer"]["global_index"] * 4]

    top_findings = await retriever.rerank_docs(
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

    answer = await safe_llm_call(llm, messages, "synthesizer")
    return {"final_answer": answer if isinstance(answer, str) else str(answer)}


def should_recurse(state: SwarmState, llm: LocalLLMClient) -> str:
    from .config import MAX_ITERATIONS, MAX_TOTAL_TOKENS, MAX_WALL_SECONDS

    elapsed = time.time() - state.get("start_time", time.time())
    if (
        elapsed > MAX_WALL_SECONDS
        or llm.tokens_used > MAX_TOTAL_TOKENS
        or state.get("iteration", 0) >= MAX_ITERATIONS
    ):
        logger.warning(
            f"Budget exhausted. Elapsed={elapsed:.0f}s "
            f"Tokens={llm.tokens_used} Iter={state.get('iteration')}"
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

    return "synthesize"
