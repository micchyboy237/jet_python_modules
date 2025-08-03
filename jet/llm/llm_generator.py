from jet.models.model_types import RoleMapping, Tool
from typing import Dict, List, Tuple, Optional, Union, TypedDict
from dataclasses import dataclass

from jet.llm.mlx.base import MLX
from jet.models.model_types import LLMModelType


class GenerationConfig(TypedDict, total=False):
    """Typed dictionary for LLM generation configuration."""
    max_tokens: int
    temperature: float
    top_p: float
    min_p: float
    min_tokens_to_keep: int
    top_k: int
    repetition_penalty: Optional[float]
    repetition_context_size: int
    xtc_probability: float
    xtc_threshold: float
    logit_bias: Optional[Union[Dict[int, float],
                               Dict[str, float], str, List[str]]]
    logprobs: int
    stop: Optional[Union[str, List[str]]]
    role_mapping: Optional[RoleMapping]
    tools: Optional[List[Tool]]
    system_prompt: Optional[str]
    log_dir: Optional[str]
    verbose: bool


class GenerateResponseResult(TypedDict):
    """Typed dictionary for the result of generate_response."""
    context: str
    response: str


DEFAULT_GENERATION_CONFIG: GenerationConfig = {
    "max_tokens": -1,
    "temperature": 0.0,
    "top_p": 1.0,
    "min_p": 0.0,
    "min_tokens_to_keep": 0,
    "top_k": 0,
    "repetition_penalty": None,
    "repetition_context_size": 20,
    "xtc_probability": 0.0,
    "xtc_threshold": 0.0,
    "logit_bias": None,
    "logprobs": -1,
    "stop": None,
    "role_mapping": None,
    "tools": None,
    "system_prompt": None,
    "log_dir": None,
    "verbose": False
}

PROMPT_TEMPLATE = """\
Context information is below.
---------------------
{context}
---------------------

Given the context information, answer the query.

Query: {query}
"""


@dataclass
class LLMConfig:
    """Configuration for LLM response generation."""
    model: LLMModelType = "qwen3-1.7b-4bit"
    max_context_length: int = 1000  # Max characters for context
    response_tone: str = "informative"  # Tone: informative, conversational, formal
    include_scores: bool = True  # Include similarity scores in context


class LLMConfigDict(TypedDict, total=False):
    """Typed dictionary for LLM configuration."""
    model: LLMModelType
    max_context_length: int
    response_tone: str
    include_scores: bool


class LLMGenerator:
    """Class for generating responses from retrieved chunks using RAG principles."""

    def __init__(self, config: Optional[Union[LLMConfig, LLMConfigDict]] = None):
        if config is None:
            self.config = LLMConfig()
        elif isinstance(config, dict):
            self.config = LLMConfig(**config)
        else:
            self.config = config

        self.llm: MLX = MLX(self.config.model)

    def _truncate_context(self, context: str, max_length: int, reserved_length: int = 0) -> str:
        """Truncate context to fit within max_length, accounting for reserved length."""
        available_length = max_length - reserved_length
        if available_length <= 0:
            return ""
        if len(context) <= available_length:
            return context
        truncated = context[:available_length]
        last_period = truncated.rfind('.')
        if last_period != -1:
            truncated = truncated[:last_period + 1]
        return truncated

    def _build_context(self, query: str, chunks: List[Tuple[str, float]]) -> str:
        """Build structured context for RAG using retrieved chunks."""
        if not chunks:
            return "No relevant information found for the query."

        context_lines = [f"Query: {query}"]
        for i, (chunk, score) in enumerate(chunks, 1):
            chunk_text = chunk.strip()
            if self.config.include_scores:
                context_lines.append(
                    f"[{i}] (Score: {score:.4f}) {chunk_text}")
            else:
                context_lines.append(f"[{i}] {chunk_text}")

        context = "\n".join(context_lines)
        return context

    def generate_response(self, query: str, chunks: List[Tuple[str, float]], template: str = PROMPT_TEMPLATE, generation_config: GenerationConfig = DEFAULT_GENERATION_CONFIG) -> GenerateResponseResult:
        """Generate a response using retrieved chunks with RAG optimization."""
        if not query:
            raise ValueError("Query cannot be empty.")

        tone = self.config.response_tone.lower()
        # if tone == "conversational":
        #     intro = f"Hey, I looked into your question: '{query}'! Here's what I found:\n"
        #     summary = f"So, to sum it up: {chunks[0][0] if chunks else 'nothing relevant.'}"
        # elif tone == "formal":
        #     intro = f"Response to the query: '{query}'\n"
        #     summary = f"Conclusion: {chunks[0][0] if chunks else 'No relevant information was retrieved.'}"
        # else:  # informative (default)
        #     intro = f"Based on the provided information for the query '{query}':\n"
        #     summary = f"In summary, {chunks[0][0] if chunks else 'no relevant information was found.'}"

        context = self._build_context(query, chunks)
        if context == "No relevant information found for the query.":
            return {
                "context": "",
                "response": "No relevant information found for the query."
            }

        # Estimate reserved length for intro and summary
        # reserved_length = len(intro) + len(summary) + 2  # +2 for newlines
        reserved_length = 0

        # Truncate context to fit within max_context_length
        truncated_context = self._truncate_context(
            context, self.config.max_context_length, reserved_length)

        prompt = template.format(query=query, context=truncated_context)
        response = self.llm.chat(prompt, **generation_config)

        return {
            "context": truncated_context,
            "response": response["content"],
        }
