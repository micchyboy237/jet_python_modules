"""Evaluate whether a full RAG context completely answers a query.
Designed for long-context evaluation where the final assembled context
may contain significant noise alongside relevant information.
Returns structured assessment including completeness flag and missing info description.
Uses grammar-constrained generation for reliable JSON output from local models.
NOTE: enable_thinking is FORCED to False because thinking tokens break grammar constraints.
"""

import json
from typing import Optional, TypedDict

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.helpers.grammar_builder import (
    build_grammar_from_schema,
    validate_grammar,
)
from jet.adapters.llama_cpp.llm_utils import chat
from jet.libs.llama_cpp.usage.chat_stream_observability import PHOENIX_URL
from jet.logger import logger


class RagRelevanceResult(TypedDict):
    """Evaluation result for full RAG context relevance."""

    is_complete: bool
    confidence: float
    missing_info: str
    is_valid: bool
    error: Optional[str]


SYSTEM_PROMPT = """\
You are an expert RAG evaluator assessing LONG CONTEXTS. Given a query and a potentially large \
assembled context, determine if the context COMPLETELY and ACCURATELY answers the query.

Evaluation Criteria:
- is_complete: true ONLY if ALL aspects of the query are fully answered by the context
- is_complete: false if ANY part remains unanswered, ambiguous, or requires external knowledge
- confidence: 0.0-1.0 indicating certainty in your assessment
- missing_info: If not complete, describe SPECIFICALLY what information is missing. \
If complete, use empty string ""

Rules for Long Contexts:
- Ignore irrelevant sections, boilerplate, and tangential information
- Focus strictly on factual completeness regarding the specific query
- Partial answers or implied information mean is_complete=false
- Be precise about WHAT is missing, not just that something is missing
- Return ONLY valid JSON matching the required schema
- Do NOT include any text outside the JSON object"""


def _build_relevance_schema() -> dict:
    """Build JSON Schema for RAG relevance evaluation response."""
    return {
        "type": "object",
        "properties": {
            "is_complete": {"type": "boolean"},
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "missing_info": {"type": "string"},
        },
        "required": ["is_complete", "confidence", "missing_info"],
        "additionalProperties": False,
    }


def _build_user_prompt(query: str, context: str) -> str:
    """Build user prompt with query and full context."""
    return f"Query: {query}\n\nFull Context:\n{context}"


def evaluate_rag_relevance(
    query: str,
    context: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    project_name: str | None = "eval-rag-relevance",
    phoenix_url: str = PHOENIX_URL,
) -> RagRelevanceResult:
    """Evaluate whether a full RAG context completely answers a query.

    Designed for long-context evaluation where the assembled context may be large
    and contain both relevant and irrelevant information. Uses grammar-constrained
    generation to ensure reliable structured output from small local models.

    Args:
        query: The user query to evaluate against.
        context: The full assembled RAG context (concatenated chunks).
        model: LLM model key. Defaults to LLM_MODEL.
        temperature: Sampling temperature (default: 0.0 for deterministic).
        max_tokens: Max tokens for the JSON response (default: 2048 for long missing_info).
        project_name: Phoenix project name for trace grouping. Set to None to disable tracing.
        phoenix_url: Phoenix server base URL for trace links.

    Returns:
        RagRelevanceResult with completeness assessment and missing info description.
    """
    resolved_model = model or LLM_MODEL

    # Input validation
    if not query.strip():
        logger.error("evaluate_rag_relevance: Query cannot be empty")
        return RagRelevanceResult(
            is_complete=False,
            confidence=0.0,
            missing_info="",
            is_valid=False,
            error="Empty query",
        )

    if not context.strip():
        logger.error("evaluate_rag_relevance: Context cannot be empty")
        return RagRelevanceResult(
            is_complete=False,
            confidence=0.0,
            missing_info="No context provided",
            is_valid=False,
            error="Empty context",
        )

    logger.info(
        f"evaluate_rag_relevance: model={resolved_model}, "
        f"query_len={len(query)}, context_len={len(context)}, "
        f"max_tokens={max_tokens}, project={project_name}"
    )

    # Build grammar from schema
    try:
        schema = _build_relevance_schema()
        grammar = build_grammar_from_schema(
            schema,
            prop_order={"is_complete": 0, "confidence": 1, "missing_info": 2},
        )
    except ValueError as e:
        error_msg = f"Grammar generation failed: {e}"
        logger.error(error_msg)
        return RagRelevanceResult(
            is_complete=False,
            confidence=0.0,
            missing_info="",
            is_valid=False,
            error=error_msg,
        )

    # Validate grammar
    validation_error = validate_grammar(grammar)
    if validation_error:
        error_msg = f"Grammar validation failed: {validation_error}"
        logger.error(error_msg)
        return RagRelevanceResult(
            is_complete=False,
            confidence=0.0,
            missing_info="",
            is_valid=False,
            error=error_msg,
        )

    logger.debug(f"Grammar OK ({len(grammar)} bytes)")

    # Prepare messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(query, context)},
    ]

    # Call LLM with grammar constraint
    try:
        result = chat(
            prompt="",
            model=resolved_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=False,  # CRITICAL: thinking breaks grammar
            project_name=project_name,
            phoenix_url=phoenix_url,
            extra_body_params={"grammar": grammar},
        )
    except Exception as e:
        error_msg = f"Chat completion failed: {e}"
        logger.error(error_msg)
        return RagRelevanceResult(
            is_complete=False,
            confidence=0.0,
            missing_info="",
            is_valid=False,
            error=error_msg,
        )

    # Parse response
    raw_output = result.content.strip()
    logger.debug(f"Raw output ({len(raw_output)} chars): '{raw_output[:300]}'")

    try:
        parsed: dict = json.loads(raw_output)
    except json.JSONDecodeError as e:
        error_msg = f"JSON parse failed despite grammar: {e}. Possible truncation."
        logger.error(error_msg)
        return RagRelevanceResult(
            is_complete=False,
            confidence=0.0,
            missing_info="",
            is_valid=False,
            error=error_msg,
        )

    # Validate parsed structure
    if not isinstance(parsed, dict):
        error_msg = f"Expected JSON object, got {type(parsed).__name__}"
        logger.error(error_msg)
        return RagRelevanceResult(
            is_complete=False,
            confidence=0.0,
            missing_info="",
            is_valid=False,
            error=error_msg,
        )

    # Extract and validate fields
    is_complete = parsed.get("is_complete", False)
    confidence = parsed.get("confidence", 0.0)
    missing_info = parsed.get("missing_info", "")

    # Type safety checks
    if not isinstance(is_complete, bool):
        logger.warning(
            f"is_complete was {type(is_complete).__name__}, coercing to bool"
        )
        is_complete = bool(is_complete)

    if not isinstance(confidence, (int, float)):
        logger.warning(f"confidence was {type(confidence).__name__}, defaulting to 0.0")
        confidence = 0.0
    else:
        confidence = max(0.0, min(1.0, float(confidence)))

    if not isinstance(missing_info, str):
        logger.warning(
            f"missing_info was {type(missing_info).__name__}, coercing to str"
        )
        missing_info = str(missing_info)

    logger.info(
        f"Evaluation complete: is_complete={is_complete}, "
        f"confidence={confidence:.2f}, missing_info_len={len(missing_info)}"
    )

    return RagRelevanceResult(
        is_complete=is_complete,
        confidence=confidence,
        missing_info=missing_info,
        is_valid=True,
        error=None,
    )


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # Single comprehensive long-context test case
    LONG_CONTEXT = """
SECTION 1: COMPANY HISTORY
Acme Corp was founded in 1985 by John Smith in Portland, Oregon. Originally a bicycle manufacturer,
the company pivoted to consumer electronics in 2003. Over the decades, Acme has won numerous awards
for design excellence and sustainability practices. The headquarters moved to Austin, Texas in 2015.

SECTION 2: PRODUCT LINE OVERVIEW
Acme currently offers three main product lines: SmartHome devices, WearableTech fitness trackers,
and EcoKitchen appliances. Each line emphasizes energy efficiency and user privacy. The SmartHome
line includes thermostats, cameras, and lighting systems compatible with major voice assistants.

SECTION 3: LEGAL DISCLAIMERS AND TERMS
This document contains forward-looking statements subject to risks and uncertainties. Actual results
may differ materially. All trademarks are property of their respective owners. Warranty periods vary
by region and product category. See acme.example.com/terms for full details. Limitation of liability
applies to all consumer products sold after January 1, 2024.

SECTION 4: Q3 2025 FINANCIAL HIGHLIGHTS
Revenue: $847 million (up 12% YoY)
Gross Margin: 34.2%
Operating Expenses: $215 million
Net Income: $78.3 million
EPS: $1.42
SmartHome segment contributed 45% of total revenue. WearableTech grew 28% but remains only 15% of revenue.
EcoKitchen declined 3% due to supply chain disruptions in Southeast Asia.

SECTION 5: SUSTAINABILITY REPORT
Acme achieved carbon neutrality in Scope 1 and 2 emissions in 2024. Water usage reduced by 18%.
Recycled packaging now used across 92% of product lines. Employee volunteer hours exceeded 50,000.
The company partnered with OceanCleanup Initiative donating 1% of EcoKitchen profits.

SECTION 6: LEADERSHIP TEAM
CEO: Maria Chen (appointed 2022)
CFO: Robert Williams
CTO: Dr. Aisha Patel
VP Engineering: Carlos Rodriguez
Board Chair: Elizabeth Thompson
"""

    TEST_QUERY = "What was Acme Corp's Q3 2025 net income, EPS, and year-over-year revenue growth rate for the WearableTech segment specifically?"

    console.print(
        "\n[bold green]RAG Relevance Evaluation — Long Context Test[/bold green]"
    )
    console.print(Panel(TEST_QUERY, title="Query", border_style="cyan"))
    console.print(
        f"[dim]Context length: {len(LONG_CONTEXT)} chars (~{len(LONG_CONTEXT.split())} words)[/dim]\n"
    )

    result = evaluate_rag_relevance(TEST_QUERY, LONG_CONTEXT)

    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("Field", style="bold", width=15)
    table.add_column("Value", style="white")

    complete_str = (
        "[bold green]✓ Complete[/bold green]"
        if result["is_complete"]
        else "[bold red]✗ Incomplete[/bold red]"
    )
    valid_str = (
        "[bold green]✓ Valid[/bold green]"
        if result["is_valid"]
        else "[bold red]✗ Invalid[/bold red]"
    )

    table.add_row("Status", complete_str)
    table.add_row("Confidence", f"{result['confidence']:.2f}")
    table.add_row("Missing Info", result["missing_info"] or "[dim](none)[/dim]")
    table.add_row("Valid Output", valid_str)
    if result["error"]:
        table.add_row("Error", f"[red]{result['error']}[/red]")

    console.print(table)
