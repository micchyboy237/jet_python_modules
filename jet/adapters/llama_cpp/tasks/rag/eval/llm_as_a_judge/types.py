# jet/adapters/llama_cpp/tasks/rag/eval/types.py
"""Shared types for RAG evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RelevanceJudgment(BaseModel):
    """Binary relevance classification for a single retrieved chunk."""

    is_relevant: bool = Field(description="Whether the chunk is relevant to the query")
    reason: str = Field(description="Brief justification for the classification")


class ClaimVerification(BaseModel):
    """Verification of a single claim against retrieved context."""

    claim: str = Field(description="The extracted claim text")
    status: str = Field(description="One of: supported, contradicted, not_mentioned")
    evidence: str = Field(description="Supporting context span or 'none'")


class EvalStage(Enum):
    OFFLINE_BENCHMARK = "offline_benchmark"
    PRE_GENERATION_GATE = "pre_generation_gate"
    PRODUCTION_ASYNC = "production_async"


@dataclass
class RAGEvaluationResult:
    """Unified result interface for all evaluation stages."""

    stage: EvalStage
    query: str
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    contextual_precision: Optional[float] = None
    contextual_recall: Optional[float] = None
    hallucination_rate: Optional[float] = None
    passed_gate: bool = True
    total_eval_tokens: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def has_critical_failure(self) -> bool:
        if self.faithfulness is not None and self.faithfulness < 0.5:
            return True
        if self.hallucination_rate is not None and self.hallucination_rate > 0.5:
            return True
        if self.contextual_precision is not None and self.contextual_precision < 0.3:
            return True
        return False
