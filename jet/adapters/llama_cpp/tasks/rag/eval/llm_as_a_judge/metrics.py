"""Metric computation using JetLLMJudge + embed_utils for semantic similarity."""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.scoring_utils import cosine_similarity

from .judge import JetLLMJudge

logger = logging.getLogger(__name__)


class RAGMetrics:
    CONTEXT_PRECISION_THRESHOLD = 0.5
    FAITHFULNESS_THRESHOLD = 0.7
    ANSWER_RELEVANCY_THRESHOLD = 0.6
    HALLUCINATION_THRESHOLD = 0.5

    def __init__(self, judge: JetLLMJudge):
        self.judge = judge

    async def compute_contextual_precision(
        self,
        query: str,
        contexts: list[str],
        *,
        session_id: str | None = None,
    ) -> tuple[float, int]:
        if not contexts:
            return 0.0, 0
        judgments = await asyncio.gather(
            *[
                self.judge.judge_chunk_relevance(query, chunk, session_id=session_id)
                for chunk in contexts
            ]
        )
        total_tokens = sum(t for _, t in judgments)
        weighted_score = 0.0
        relevant_count = 0
        for i, (judgment, _) in enumerate(judgments):
            if judgment.is_relevant:
                relevant_count += 1
                weighted_score += relevant_count / (i + 1)
        return weighted_score / len(contexts), total_tokens

    async def compute_faithfulness(
        self,
        response: str,
        contexts: list[str],
        *,
        session_id: str | None = None,
    ) -> tuple[float, float, int]:
        context_text = "\n---\n".join(contexts)
        claims, extract_tokens = await self.judge.extract_claims(
            response, session_id=session_id
        )
        if not claims:
            return 1.0, 0.0, extract_tokens
        verifications, verify_tokens = await self.judge.verify_claims(
            claims, context_text, session_id=session_id
        )
        total_tokens = extract_tokens + verify_tokens
        if not verifications:
            return 0.0, 1.0, total_tokens
        supported = sum(1 for v in verifications if v["status"] == "supported")
        contradicted = sum(1 for v in verifications if v["status"] == "contradicted")
        not_mentioned = sum(1 for v in verifications if v["status"] == "not_mentioned")
        faithfulness = supported / len(verifications)
        hallucination_rate = (contradicted + not_mentioned) / len(verifications)
        return faithfulness, hallucination_rate, total_tokens

    async def compute_answer_relevancy(
        self,
        query: str,
        response: str,
        *,
        session_id: str | None = None,
    ) -> tuple[float, int]:
        """Semantic similarity via embeddings instead of lexical overlap."""
        questions, tokens = await self.judge.generate_reverse_questions(
            response, session_id=session_id
        )
        if not questions:
            return 0.0, tokens
        all_texts = [query] + questions
        embeddings = embed(all_texts, return_format="numpy")
        query_emb = embeddings[0]
        question_embs = embeddings[1:]
        similarities = [cosine_similarity(query_emb, q_emb) for q_emb in question_embs]
        relevancy = float(np.mean(similarities))
        return relevancy, tokens
