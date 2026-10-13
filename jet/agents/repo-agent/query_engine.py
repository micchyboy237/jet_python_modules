"""
Main orchestration layer: manages context assembly and LLM interaction.
"""

from typing import Dict, List

import aiohttp
from config import servers
from retriever import Retriever
from utils import estimate_tokens


class QueryEngine:
    def __init__(self):
        self.retriever = Retriever()

    async def query(self, question: str) -> Dict:
        # 1. Retrieve Context
        results = await self.retriever.retrieve(question)

        if not results:
            return {
                "answer": "I couldn't find relevant information in the codebase.",
                "sources": [],
            }

        # 2. Assemble Context with Token Budgeting
        # Reserve 2048 tokens for system prompt + response generation
        max_context_tokens = 16384 - 2048
        context_str, sources = self._build_context(results, max_context_tokens)

        # 3. Generate Answer
        answer = await self._call_llm(question, context_str)

        return {"answer": answer, "sources": sources}

    def _build_context(self, results: List[Dict], max_tokens: int) -> tuple:
        context_parts = []
        sources = []
        current_tokens = 0

        for res in results:
            source_info = f"File: {res['file_path']} (Lines {res['start_line']}-{res['end_line']})"
            snippet = f"```\n{res['content']}\n```"
            block = f"[{source_info}]\n{snippet}\n"

            block_tokens = estimate_tokens(block)
            if current_tokens + block_tokens > max_tokens:
                break

            context_parts.append(block)
            sources.append(
                {
                    "file": res["file_path"],
                    "lines": f"{res['start_line']}-{res['end_line']}",
                    "score": res.get("rerank_score", res.get("_distance", 0)),
                }
            )
            current_tokens += block_tokens

        return "\n".join(context_parts), sources

    async def _call_llm(self, question: str, context: str) -> str:
        prompt = f"""You are a helpful coding assistant. Answer based ONLY on the provided context.

Context:
{context}

Question: {question}

Answer:"""

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "prompt": prompt,
                    "temperature": 0.2,
                    "max_tokens": 1024,
                    "stop": ["</s>", "Question:"],
                }
                async with session.post(
                    f"{servers.LLM_URL}/completions", json=payload
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["text"].strip()
                    else:
                        return "Error: LLM server returned an error."
        except Exception as e:
            return f"Error connecting to LLM: {e}"
