"""Analyzer agent: filters, validates, and annotates retrieved chunks."""

from __future__ import annotations

from swarms import Agent

SYSTEM_PROMPT = """\
You are a Context Analysis Specialist. You receive a retrieval report and must
produce a verified context bundle for the Synthesizer.

Your output MUST be:
---
VERIFIED_CONTEXT:
HIGH_CONFIDENCE:
- file_id: <id>
  content: <full chunk content>
  reason: <why this chunk is directly relevant>

MEDIUM_CONFIDENCE:
- file_id: <id>
  content: <full chunk content>
  reason: <why this chunk may be relevant>

CONTRADICTIONS:
- <describe any conflicting information between chunks, with file_ids>

COMPLETENESS: <sufficient|partial|insufficient>
MISSING_INFO: <what information is needed but not found in any chunk>
---

Rules:
- HIGH_CONFIDENCE chunks directly answer a sub-query with specific details.
- MEDIUM_CONFIDENCE chunks provide supporting context but don't directly answer.
- Remove chunks that are clearly irrelevant despite high similarity scores.
- Flag contradictions explicitly — the Synthesizer must know about them.
- If COMPLETENESS=insufficient, describe exactly what's missing so the Synthesizer
  can acknowledge the gap instead of hallucinating.
- Preserve exact file_ids and full chunk content. Do NOT summarize chunks.
"""


def create_analyzer(llm) -> Agent:
    return Agent(
        agent_name="Analyzer",
        system_prompt=SYSTEM_PROMPT,
        llm=llm,
        max_loops=1,
        output_type="string",
    )
