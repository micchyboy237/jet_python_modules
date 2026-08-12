"""Evaluate whether a full RAG context completely answers a query.
Designed for long-context evaluation where the final assembled context
may contain significant noise alongside relevant information.
Returns structured assessment with decomposed_queries, completed_info,
and missing_info as lists of short texts for programmatic consumption.
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

    decomposed_queries: list[str]
    is_complete: bool
    confidence: float
    completed_info: list[str]
    missing_info: list[str]
    is_valid: bool
    error: Optional[str]


SYSTEM_PROMPT = """\
You are an expert RAG evaluator assessing LONG CONTEXTS. Given a query and a potentially large \
assembled context, determine if the context COMPLETELY and ACCURATELY answers the query.

STEP 1 — DECOMPOSE: Break the query into atomic sub-questions. Each sub-question MUST include \
its full scope (entity, metric, time period). List them in decomposed_queries. \
Example: "WearableTech segment Q3 2025 net income" is ONE sub-question, NOT separate parts.

STEP 2 — EXTRACT: For EACH sub-question, search the context for a value that matches the FULL SCOPE. \
A company-wide value does NOT satisfy a segment-specific sub-question. Scope must match exactly.

STEP 3 — CLASSIFY: Place each sub-question's result in exactly one list:
- completed_info: Sub-questions where a scope-matching value was found. Include the exact value.
- missing_info: Sub-questions where NO scope-matching value exists in the context.

Output Format:
- decomposed_queries: List of atomic sub-questions derived from the original query
- is_complete: true ONLY if missing_info is empty
- confidence: 0.0-1.0 indicating certainty in your assessment
- completed_info: List of short statements. Each MUST name the full scope AND the exact value found.
- missing_info: List of short statements. Each MUST name the full scope and state what is absent.

Rules:
- Every sub-question in decomposed_queries must appear in EXACTLY ONE of completed_info or missing_info
- SCOPE MATCHING IS STRICT: "WearableTech EPS" ≠ "Company EPS". Never substitute.
- Keep each list item concise (under 30 words)
- Extract values from bullets (•), tables, and structured formats accurately
- Ignore irrelevant context sections
- Return ONLY valid JSON matching the required schema
- Do NOT include any text outside the JSON object"""


def _build_relevance_schema() -> dict:
    """Build JSON Schema for RAG relevance evaluation response."""
    return {
        "type": "object",
        "properties": {
            "decomposed_queries": {
                "type": "array",
                "items": {"type": "string"},
            },
            "is_complete": {"type": "boolean"},
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "completed_info": {
                "type": "array",
                "items": {"type": "string"},
            },
            "missing_info": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "decomposed_queries",
            "is_complete",
            "confidence",
            "completed_info",
            "missing_info",
        ],
        "additionalProperties": False,
    }


def _build_user_prompt(query: str, context: str) -> str:
    """Build user prompt with query and full context."""
    return f"Query: {query}\n\nFull Context:\n{context}"


def _validate_string_list(raw: object, field_name: str) -> list[str]:
    """Ensure parsed value is a list of strings, coercing/filtering as needed."""
    if not isinstance(raw, list):
        logger.warning(f"{field_name} was {type(raw).__name__}, defaulting to []")
        return []
    validated: list[str] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            validated.append(item)
        else:
            logger.warning(
                f"{field_name}[{i}] was {type(item).__name__}, coercing to str"
            )
            validated.append(str(item))
    return validated


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
    and contain both relevant and irrelevant information. Returns structured
    decomposed_queries, completed_info, and missing_info as lists of short texts
    for programmatic consumption.

    Args:
        query: The user query to evaluate against.
        context: The full assembled RAG context (concatenated chunks).
        model: LLM model key. Defaults to LLM_MODEL.
        temperature: Sampling temperature (default: 0.0 for deterministic).
        max_tokens: Max tokens for the JSON response (default: 2048 for list outputs).
        project_name: Phoenix project name for trace grouping. Set to None to disable tracing.
        phoenix_url: Phoenix server base URL for trace links.

    Returns:
        RagRelevanceResult with decomposition, completeness assessment, and info lists.
    """
    resolved_model = model or LLM_MODEL

    # Input validation
    if not query.strip():
        logger.error("evaluate_rag_relevance: Query cannot be empty")
        return RagRelevanceResult(
            decomposed_queries=[],
            is_complete=False,
            confidence=0.0,
            completed_info=[],
            missing_info=[],
            is_valid=False,
            error="Empty query",
        )

    if not context.strip():
        logger.error("evaluate_rag_relevance: Context cannot be empty")
        return RagRelevanceResult(
            decomposed_queries=[],
            is_complete=False,
            confidence=0.0,
            completed_info=[],
            missing_info=["No context provided"],
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
            prop_order={
                "decomposed_queries": 0,
                "is_complete": 1,
                "confidence": 2,
                "completed_info": 3,
                "missing_info": 4,
            },
        )
    except ValueError as e:
        error_msg = f"Grammar generation failed: {e}"
        logger.error(error_msg)
        return RagRelevanceResult(
            decomposed_queries=[],
            is_complete=False,
            confidence=0.0,
            completed_info=[],
            missing_info=[],
            is_valid=False,
            error=error_msg,
        )

    # Validate grammar
    validation_error = validate_grammar(grammar)
    if validation_error:
        error_msg = f"Grammar validation failed: {validation_error}"
        logger.error(error_msg)
        return RagRelevanceResult(
            decomposed_queries=[],
            is_complete=False,
            confidence=0.0,
            completed_info=[],
            missing_info=[],
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
            decomposed_queries=[],
            is_complete=False,
            confidence=0.0,
            completed_info=[],
            missing_info=[],
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
            decomposed_queries=[],
            is_complete=False,
            confidence=0.0,
            completed_info=[],
            missing_info=[],
            is_valid=False,
            error=error_msg,
        )

    # Validate parsed structure
    if not isinstance(parsed, dict):
        error_msg = f"Expected JSON object, got {type(parsed).__name__}"
        logger.error(error_msg)
        return RagRelevanceResult(
            decomposed_queries=[],
            is_complete=False,
            confidence=0.0,
            completed_info=[],
            missing_info=[],
            is_valid=False,
            error=error_msg,
        )

    # Extract and validate all fields
    decomposed_queries = _validate_string_list(
        parsed.get("decomposed_queries"), "decomposed_queries"
    )
    is_complete = parsed.get("is_complete", False)
    confidence = parsed.get("confidence", 0.0)
    completed_info = _validate_string_list(
        parsed.get("completed_info"), "completed_info"
    )
    missing_info = _validate_string_list(parsed.get("missing_info"), "missing_info")

    # Type safety checks for scalar fields
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

    logger.info(
        f"Evaluation complete: is_complete={is_complete}, "
        f"confidence={confidence:.2f}, "
        f"decomposed={len(decomposed_queries)}, "
        f"completed={len(completed_info)}, missing={len(missing_info)}"
    )

    return RagRelevanceResult(
        decomposed_queries=decomposed_queries,
        is_complete=is_complete,
        confidence=confidence,
        completed_info=completed_info,
        missing_info=missing_info,
        is_valid=True,
        error=None,
    )


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # Long context that PARTIALLY answers the query — demonstrates incomplete RAG evaluation
    # Contains company-wide financials but lacks segment-specific EPS and net income
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
Total Company Revenue: $847 million (up 12% YoY)
Gross Margin: 34.2%
Operating Expenses: $215 million
Total Net Income: $78.3 million
Total EPS: $1.42

Segment Breakdown — Q3 2025:
• SmartHome: Revenue $381M (45% of total), YoY Revenue Growth +8%
• WearableTech: Revenue $127M (15% of total), YoY Revenue Growth +28%
• EcoKitchen: Revenue $339M (40% of total), YoY Revenue Growth -3%

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

    TEST_QUERY = (
        "What was Acme Corp's Q3 2025 net income, EPS, and year-over-year "
        "revenue growth rate for the WearableTech segment specifically?"
    )

    console.print(
        "\n[bold green]RAG Relevance Evaluation — Incomplete Context Test[/bold green]"
    )
    console.print(Panel(TEST_QUERY, title="Query", border_style="cyan"))
    console.print(
        f"[dim]Context length: {len(LONG_CONTEXT)} chars "
        f"(~{len(LONG_CONTEXT.split())} words)[/dim]\n"
    )

    result = evaluate_rag_relevance(TEST_QUERY, LONG_CONTEXT)

    # Show decomposed queries first
    if result["decomposed_queries"]:
        console.print("[bold cyan]Decomposed Queries:[/bold cyan]")
        for i, dq in enumerate(result["decomposed_queries"], 1):
            console.print(f"  {i}. {dq}")
        console.print()
    else:
        console.print("[dim]Decomposed Queries: (none)[/dim]\n")

    # Main result table
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
    table.add_row("Valid Output", valid_str)
    if result["error"]:
        table.add_row("Error", f"[red]{result['error']}[/red]")

    console.print(table)

    # Completed info list
    if result["completed_info"]:
        console.print("\n[bold green]✓ Completed Info:[/bold green]")
        for i, item in enumerate(result["completed_info"], 1):
            console.print(f"  {i}. {item}")
    else:
        console.print("\n[dim]✓ Completed Info: (none)[/dim]")

    # Missing info list
    if result["missing_info"]:
        console.print("\n[bold red]✗ Missing Info:[/bold red]")
        for i, item in enumerate(result["missing_info"], 1):
            console.print(f"  {i}. {item}")
    else:
        console.print(
            "\n[bold green]✗ Missing Info: (none — context is complete)[/bold green]"
        )
