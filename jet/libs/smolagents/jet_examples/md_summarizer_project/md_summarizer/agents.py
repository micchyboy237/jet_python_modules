"""Prompt templates and the agent roles used by the pipeline.

All roles share one underlying model (via an LLMClient) -- what makes them
distinct "agents" is the system prompt and where in the tree each one runs.
Summaries are kept as extractive bullet points between levels (rather than
free prose) specifically to limit hallucination drift compounding across
several rounds of recursive merging on a small model.
"""

import logging
import random
from typing import List

from .chunking import chunk_markdown
from .llm_client import LLMClient
from .reduce_utils import hierarchical_reduce

logger = logging.getLogger("md_summarizer.agents")

MAPPER_SYSTEM_PROMPT = (
    "You are the Mapper agent in a document summarization pipeline. "
    "Read the markdown segment and extract the key facts as short bullet points. "
    "Be extractive, not creative: only include information explicitly stated in the text. "
    "Do not add commentary, opinions, or information not present in the text. "
    "Output 3-8 bullet points, each under 20 words."
)

REDUCER_SYSTEM_PROMPT = (
    "You are the Reducer agent in a document summarization pipeline. "
    "You will be given several bullet-point summaries from related documents or sections. "
    "Merge them into a single, deduplicated bullet-point list that preserves every distinct "
    "fact. Do not invent connections or facts that are not present in the input. "
    "Output 5-10 bullet points, each under 20 words."
)

SYNTHESIZER_SYSTEM_PROMPT = (
    "You are the Synthesizer agent, the final step in a document summarization pipeline. "
    "You will be given bullet-point summaries covering an entire project's documentation. "
    "Write a short, coherent prose digest (3-6 sentences per major area) that a new team "
    "member could read to understand the project. Only use information present in the input; "
    "do not speculate."
)

VERIFIER_SYSTEM_PROMPT = (
    "You are the Verifier agent. You will be given a final digest and a list of claims drawn "
    "from original source summaries. For EACH claim, answer only SUPPORTED, UNSUPPORTED, or "
    "PARTIALLY SUPPORTED, with a one-sentence reason. Be strict: if the digest does not clearly "
    "reflect the claim, mark it UNSUPPORTED or PARTIALLY SUPPORTED."
)


def summarize_file(
    llm: LLMClient,
    file_path: str,
    text: str,
    token_budget: int,
    max_tokens_out: int,
    temperature: float,
) -> str:
    """Mapper role: turn one markdown file into a single bullet-point summary,
    chunking internally first if the file itself is too large for one call."""
    chunks = chunk_markdown(text, llm.count_tokens, token_budget, source_label=file_path)
    if not chunks:
        return ""

    def map_one(chunk_text: str) -> str:
        return llm.complete(MAPPER_SYSTEM_PROMPT, chunk_text, max_tokens_out, temperature)

    def reduce_batch(batch: List[str]) -> str:
        joined = "\n\n---\n\n".join(batch)
        return llm.complete(REDUCER_SYSTEM_PROMPT, joined, max_tokens_out, temperature)

    logger.info("[%s] mapping %d chunk(s)", file_path, len(chunks))
    chunk_summaries = [map_one(c.text) for c in chunks]

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    return hierarchical_reduce(
        chunk_summaries, reduce_batch, llm.count_tokens, token_budget, node_label=file_path
    )


def reduce_folder(
    llm: LLMClient,
    folder_label: str,
    child_summaries: List[str],
    token_budget: int,
    max_tokens_out: int,
    temperature: float,
) -> str:
    """Reducer role: merge a folder's children (file summaries and/or
    subfolder summaries) into one folder-level summary, recursing internally
    if there are too many/too-large children for one call."""
    if not child_summaries:
        return ""
    if len(child_summaries) == 1:
        return child_summaries[0]

    def reduce_batch(batch: List[str]) -> str:
        joined = "\n\n---\n\n".join(batch)
        return llm.complete(REDUCER_SYSTEM_PROMPT, joined, max_tokens_out, temperature)

    return hierarchical_reduce(
        child_summaries, reduce_batch, llm.count_tokens, token_budget, node_label=folder_label
    )


def synthesize_root(
    llm: LLMClient,
    top_level_summaries: List[str],
    token_budget: int,
    max_tokens_out: int,
    temperature: float,
) -> str:
    """Synthesizer role: produce the final coherent digest from the root's
    immediate children summaries."""

    def reduce_batch(batch: List[str]) -> str:
        joined = "\n\n---\n\n".join(batch)
        return llm.complete(SYNTHESIZER_SYSTEM_PROMPT, joined, max_tokens_out, temperature)

    return hierarchical_reduce(
        top_level_summaries, reduce_batch, llm.count_tokens, token_budget, node_label="root-digest"
    )


def verify_digest(
    llm: LLMClient,
    digest: str,
    leaf_summaries: List[str],
    token_budget: int,
    max_tokens_out: int,
    temperature: float,
    sample_size: int = 3,
) -> str:
    """Verifier role: spot-check a random sample of leaf-level facts against
    the final digest for a cheap faithfulness signal. This is a smoke test for
    drift, not an exhaustive audit -- see the README for why full verification
    isn't worth it on every fact at this model size."""
    sample = random.sample(leaf_summaries, k=min(sample_size, len(leaf_summaries)))
    claims_block = "\n".join(f"- {s}" for s in sample)
    prompt = f"Final digest:\n{digest}\n\nClaims to check:\n{claims_block}"

    if llm.count_tokens(prompt) > token_budget:
        logger.warning("verifier prompt exceeds token budget; trimming claim sample")
        sample = sample[: max(1, len(sample) // 2)]
        claims_block = "\n".join(f"- {s}" for s in sample)
        prompt = f"Final digest:\n{digest}\n\nClaims to check:\n{claims_block}"

    return llm.complete(VERIFIER_SYSTEM_PROMPT, prompt, max_tokens_out, temperature)
