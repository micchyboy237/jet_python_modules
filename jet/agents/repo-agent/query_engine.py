"""
Main orchestration layer: manages context assembly and LLM interaction.
"""

from typing import Dict, List

import aiohttp
from config import servers
from jet.adapters.llama_cpp.token_utils import count_tokens
from retriever import Retriever
from utils import logger


class QueryEngine:
    def __init__(self):
        self.retriever = Retriever()

    async def query(self, question: str) -> Dict:
        results = await self.retriever.retrieve(question)
        if not results:
            return {
                "answer": "I couldn't find relevant information in the codebase.",
                "sources": [],
            }

        max_context_tokens = 16384 - 2048
        context_str, sources = self._build_context(results, max_context_tokens)
        answer = await self._call_llm(question, context_str)
        return {"answer": answer, "sources": sources}

    def _build_context(self, results: List[Dict], max_tokens: int) -> tuple:
        context_parts = []
        sources = []
        current_tokens = 0

        for idx, res in enumerate(results):
            source_info = f"File: {res['file_path']} (Lines {res['start_line']}-{res['end_line']})"
            snippet = f"```\n{res['content']}\n```"
            block = f"[{source_info}]\n{snippet}\n"

            # Use accurate token counting
            block_tokens = count_tokens(block)

            if current_tokens + block_tokens > max_tokens:
                logger.warning(
                    f"Context limit reached ({current_tokens}/{max_tokens} tokens). "
                    f"Skipping remaining {len(results) - idx} chunks."
                )
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

        logger.info(
            f"Built context with {len(context_parts)} chunks, {current_tokens} tokens"
        )
        return "\n".join(context_parts), sources

    async def _call_llm(self, question: str, context: str) -> str:
        prompt = f"""You are a helpful coding assistant. Answer based ONLY on the provided context.
If the context doesn't contain enough information, say so clearly.
Do not make up code or APIs that aren't shown in the context.

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
                    "stop": ["</s>", "Question:", "\n\n"],
                }
                async with session.post(
                    f"{servers.LLM_URL}/completions", json=payload
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        answer = data["choices"][0]["text"].strip()
                        logger.debug(f"LLM response length: {len(answer)} chars")
                        return answer
                    else:
                        error_text = await resp.text()
                        logger.error(f"LLM API error {resp.status}: {error_text}")
                        return "Error: LLM server returned an error."
        except Exception as e:
            logger.error(f"Error connecting to LLM: {e}")
            return f"Error connecting to LLM: {e}"
