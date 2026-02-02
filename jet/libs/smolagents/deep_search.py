# deep_search.py

import logging
import re
import shutil
from pathlib import Path
from typing import Any

# ---- Added: Accurate token counter import
from jet.adapters.llama_cpp.tokens import count_tokens
from jet.adapters.llama_cpp.types import LLAMACPP_LLM_KEYS
from jet.libs.smolagents.custom_models import OpenAIModel
from jet.libs.smolagents.step_callbacks import save_step_state
from jet.libs.smolagents.step_callbacks.memory_window import memory_window_limiter
from jet.libs.smolagents.tools.visit_webpage_tool import VisitWebpageTool
from jet.libs.smolagents.tools.web_search_tool import WebSearchTool
from rich.console import Console

# ────────────────────────────────────────────────
# Logging & rich setup
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from smolagents import (
    CodeAgent,
    LogLevel,
    ToolCallingAgent,
    tool,
)

# ---- Added: Token threshold constants
MAX_SAFE_PROMPT_TOKENS = 6500  # adjust for your model's window/context
WARNING_THRESHOLD_TOKENS = 5800
FORCE_FINAL_THRESHOLD_TOKENS = 7200  # approach model limit/cutoff
VISITED_PAGE_OUTPUT_TOKENS = 5000

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("deep_research")


def create_local_model(
    temperature: float = 0.4,
    max_tokens: int | None = 4096,
    model_id: LLAMACPP_LLM_KEYS = "qwen3-instruct-2507:4b",
    agent_name: str | None = None,
) -> OpenAIModel:
    """Factory for creating consistently configured local llama.cpp model."""
    return OpenAIModel(
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        agent_name=agent_name,
    )


@tool
def extract_key_facts(content: str, query: str) -> str:
    """
    Extracts and returns the most relevant lines/facts from page content for a given query.

    Args:
        content: The markdown or text content of a webpage to analyze.
        query: The original user question or search phrase to match against.

    Returns:
        A short string containing relevant excerpts or a message if nothing matched.
    """
    # In a real system you could use a small LLM call here.
    # For now we do simple heuristic extraction.
    lines = content.split("\n")
    relevant = []
    query_lower = query.lower()

    for line in lines:
        if len(line.strip()) < 15:
            continue
        if any(term in line.lower() for term in query_lower.split()):
            relevant.append(line.strip())

    if not relevant:
        return "No clearly relevant information found for the query on this page."

    summary = "\n".join(relevant[:12])  # limit to avoid token explosion
    return f"Relevant excerpts:\n{summary}"


# ────────────────────────────────────────────────
# Agents
# ────────────────────────────────────────────────


def create_web_research_agent():
    model = create_local_model(agent_name="web_research_agent")
    return ToolCallingAgent(
        tools=[
            WebSearchTool(
                embed_model=model.model_id,
            ),
            VisitWebpageTool(
                embed_model=model.model_id, max_output_length=VISITED_PAGE_OUTPUT_TOKENS
            ),
            # extract_key_facts,
        ],
        model=model,
        name="web_research_agent",
        step_callbacks=[
            save_step_state("web_research_agent"),
            memory_window_limiter(max_recent_steps=4),  # ← add here
        ],
        description=(
            "A specialized agent that performs web searches, visits pages, "
            "and extracts relevant information. Use this agent when you need "
            "fresh information from the internet or need to dive deeper into "
            "specific sources. Provide clear, specific instructions."
        ),
        max_steps=12,
        provide_run_summary=True,  # ← important for token saving
        verbosity_level=LogLevel.DEBUG,
    )


def create_evidence_evaluator_agent():
    model = create_local_model(agent_name="evidence_evaluator_agent")
    # Slightly better prompt inside description (helps model output parseable text)
    description = (
        "Evaluates collected evidence. Output format:\n"
        "CONFIDENCE: <0-10 integer>\nSUFFICIENT: yes/no\nMISSING: ...\nCONTRADICTIONS: ...\nSOURCES: url1, url2, ...\nREASON: ..."
    )
    return ToolCallingAgent(
        tools=[],
        model=model,
        name="evidence_evaluator",
        step_callbacks=[save_step_state("evidence_evaluator")],
        description=description,
        max_steps=4,
        provide_run_summary=True,
        verbosity_level=LogLevel.DEBUG,
    )


def create_query_refiner():
    model = create_local_model(agent_name="query_refiner")
    return ToolCallingAgent(
        tools=[],
        model=model,
        name="query_refiner",
        step_callbacks=[save_step_state("query_refiner")],
        description=(
            "Reformulates the original user query into 1–3 more precise, "
            "effective search queries when initial results are poor, "
            "ambiguous or missing key aspects. "
            "Returns a list of improved queries as bullet points."
        ),
        max_steps=5,
        provide_run_summary=True,
        verbosity_level=LogLevel.DEBUG,
    )


def create_final_formatter():
    model = create_local_model(agent_name="final_formatter")
    return ToolCallingAgent(
        tools=[],
        model=model,
        name="final_formatter",
        step_callbacks=[save_step_state("final_formatter")],
        description=(
            "Takes raw research output, evidence summaries and sources, "
            "then produces clean, well-structured markdown with inline "
            "citations [1], [2]… and a References section at the bottom."
        ),
        max_steps=4,
        provide_run_summary=True,
        verbosity_level=LogLevel.DEBUG,
    )


# ────────────────────────────────────────────────
# Update manager creation


def create_deep_research_manager():
    web_agent = create_web_research_agent()
    eval_agent = create_evidence_evaluator_agent()
    refiner_agent = create_query_refiner()
    formatter_agent = create_final_formatter()

    model = create_local_model(agent_name="deep_research_manager")
    manager = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[
            web_agent,
            eval_agent,
            refiner_agent,
            formatter_agent,
        ],
        name="deep_research_manager",
        step_callbacks=[
            save_step_state("deep_research_manager"),
            memory_window_limiter(max_recent_steps=6),  # slightly larger for manager
        ],
        description=(
            "Top-level research coordinator. Uses specialized sub-agents to "
            "deeply investigate a question, gather evidence, evaluate it, "
            "refine queries when stuck, and produce clean final answers."
        ),
        max_steps=20,
        additional_authorized_imports=["datetime"],
        verbosity_level=LogLevel.DEBUG,
    )

    return manager


# ────────────────────────────────────────────────
# Main loop with rich UI + memory


def run_deep_search(
    query: str,
    model_id: str | None = None,
    temperature: float = 0.4,
    max_research_rounds: int = 5,
    min_confidence: int = 7,
    history_window: int = 4,  # Controls how many recent rounds to show in full detail
) -> str:
    manager = create_deep_research_manager()

    round_summaries: list[str] = []
    evidence_accumulator: list[dict[str, Any]] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[bold]{task.completed}/{task.total} rounds"),
    )

    # --- Bonus A: Make evidence summary token aware ---
    def build_evidence_summary(max_tokens: int = 1400) -> str:
        if not evidence_accumulator:
            return "(no evidence collected yet)"

        lines = []
        current_tokens = 0
        added_omitted = False

        # Newest first, truncate oldest if needed
        for ev in reversed(evidence_accumulator):
            srcs = ", ".join(ev.get("sources", [])[:2])
            facts = ev.get("raw_output", "")[:220].replace("\n", " ").strip()
            line = (
                f"Round {ev['round']} (conf {ev.get('confidence', '?')}, "
                f"suff {ev.get('sufficient', '?')}): {facts}"
            )
            if srcs:
                line += f" [src: {srcs}]"

            line_tokens = count_tokens(line, model=model_id) + 10  # small buffer

            if current_tokens + line_tokens > max_tokens:
                if not added_omitted:
                    lines.append("... older evidence omitted for context limit ...")
                    added_omitted = True
                continue

            lines.append(line)
            current_tokens += line_tokens

        lines.reverse()
        return "\n".join(lines) if lines else "(no usable evidence)"

    research_plan_template = """
You are coordinating thorough, critical web research.

Original query: {query}

Current round summaries (previous knowledge):
{history}

Current compact evidence memory:
{evidence_summary}

Instructions:
- You have agents: query_refiner, web_research_agent, evidence_evaluator, final_formatter
- Use query_refiner when first results are weak/ambiguous
- Use web_research_agent to gather evidence
- After gathering → always call evidence_evaluator
- evidence_evaluator returns: confidence (0-10), sufficient (yes/no), missing aspects, contradictions
- If confidence >= {min_conf} and no major gaps → call final_formatter → output FINAL ANSWER
- If not → continue (max {max_rounds} rounds)
- Be critical about conflicts, source dates, authority
- Keep track of good URLs/sources

Start now.
""".strip()

    with Live(
        Panel("Starting deep research…", title="Deep Research", border_style="blue"),
        refresh_per_second=4,
        console=console,
    ) as live:
        task = progress.add_task("[cyan]Research progress", total=max_research_rounds)

        round_num = 0
        result = ""
        final_answer = None

        while round_num < max_research_rounds:
            round_num += 1
            logger.info(f"[Round {round_num}] Starting...")

            history_text = (
                "\n".join(round_summaries)
                if round_summaries
                else "(no previous rounds)"
            )
            evidence_text = build_evidence_summary()

            current_plan = research_plan_template.format(
                query=query,
                history=history_text,
                evidence_summary=evidence_text,
                min_conf=min_confidence,
                max_rounds=max_research_rounds,
            )

            result = manager.run(current_plan)

            # --- Accurate token counting (replacing old estimation) ---
            next_prompt_tokens = count_tokens(
                current_plan, model=model_id, add_special_tokens=False
            )

            output_tokens = count_tokens(
                result, model=model_id, add_special_tokens=False
            )

            total_approx_for_next = next_prompt_tokens + output_tokens // 2 + 800

            logger.info(
                f"[Round {round_num}] Tokens: prompt={next_prompt_tokens:,}, "
                f"output={output_tokens:,} → est. next={total_approx_for_next:,}"
            )

            if total_approx_for_next > FORCE_FINAL_THRESHOLD_TOKENS:
                logger.warning(
                    f"[Round {round_num}] Approaching context limit ({total_approx_for_next:,} est.) → "
                    f"forcing final format"
                )
                # Optionally force-call final_formatter here
                break

            if total_approx_for_next > WARNING_THRESHOLD_TOKENS:
                logger.warning(
                    f"[Round {round_num}] High token usage ({total_approx_for_next:,} est.) — "
                    f"consider concluding soon"
                )

            # --- End accurate token counting ---

            # Simple best-effort parsing of evaluator output
            confidence = None
            sufficient = "no"
            missing = ""
            contradictions = ""
            sources: list[str] = re.findall(r"https?://[^\s,]+", result)

            for line in result.split("\n"):
                line = line.strip()
                if line.startswith("CONFIDENCE:"):
                    confidence = int(line.split(":", 1)[1].strip().split("/")[0])
                elif line.startswith("SUFFICIENT:"):
                    sufficient = "yes" if "yes" in line.lower() else "no"
                elif line.startswith("MISSING:"):
                    missing = line.split(":", 1)[1].strip()
                elif line.startswith("CONTRADICTIONS:"):
                    contradictions = line.split(":", 1)[1].strip()

            # --- Bonus B: Summarize very long results before storing ---
            if count_tokens(result, model=model_id) > 1800:
                # Truncate aggressively or call extract_key_facts if re-enabled
                result_summary = (
                    result[:900] + " [...] [truncated — full details in sources]"
                )
                evidence_accumulator.append(
                    {
                        "round": round_num,
                        "confidence": confidence,
                        "sufficient": sufficient,
                        "missing": missing,
                        "contradictions": contradictions,
                        "sources": sources,
                        "raw_output": result_summary,
                    }
                )
            else:
                evidence_accumulator.append(
                    {
                        "round": round_num,
                        "confidence": confidence,
                        "sufficient": sufficient,
                        "missing": missing,
                        "contradictions": contradictions,
                        "sources": sources,
                        "raw_output": result[:360] + "...",
                    }
                )
            # --- End Bonus B ---

            round_summaries.append(
                f"Round {round_num}: {result[:180]}... "
                f"(conf: {confidence}, suff: {sufficient})"
            )

            live.update(
                Panel(
                    progress,
                    title=f"Deep Research – {query[:50]}… (round {round_num})",
                    border_style="green" if sufficient == "yes" else "yellow",
                )
            )

            progress.advance(task)

            if sufficient == "yes" and (confidence or 0) >= min_confidence:
                logger.info(
                    f"[Round {round_num}] High confidence — moving to final format"
                )
                break

        # ── Final formatting step ──────────────────────────────
        if evidence_accumulator:
            logger.info("Calling final_formatter...")

            formatter_input = f"""

Original query: {query}

Latest evidence summary:
{build_evidence_summary()}

Most recent raw outputs (very abbreviated):
{chr(10).join([f"R{e['round']}: {e['raw_output'][:140]}..." for e in evidence_accumulator[-3:]])}

Create clean markdown answer with inline citations [1], [2]...
Include References section at bottom with URLs.

Be concise. Focus on most important findings.
"""
            final_formatter = manager.managed_agents[-1]
            final_text = final_formatter.run(formatter_input)
        else:
            final_text = result

        if "FINAL ANSWER" not in final_text:
            final_text = f"{final_text}\n\n[Note: no explicit final answer marker — using last output]"

        console.print("\n[bold green]Research completed.[/bold green]")
        console.print(Panel(final_text, title="Final Answer", border_style="green"))

        # Optional: show evidence table at end
        if evidence_accumulator:
            table = Table(title="Evidence Summary")
            table.add_column("Round", justify="right")
            table.add_column("Conf")
            table.add_column("Suff")
            table.add_column("Sources")
            for ev in evidence_accumulator:
                srcs = ", ".join(ev["sources"][:2]) + (
                    "..." if len(ev["sources"]) > 2 else ""
                )
                table.add_row(
                    str(ev["round"]),
                    str(ev.get("confidence", "?")),
                    ev["sufficient"],
                    srcs,
                )
            console.print(table)

    return final_text


# ────────────────────────────────────────────────
# Usage example
# ────────────────────────────────────────────────

if __name__ == "__main__":
    query = "What are the main differences in architecture and training between DeepSeek-R1 and Qwen 3 models?"

    answer = run_deep_search(
        query=query,
        # model_id="meta-llama/Llama-4-70B-Instruct",  # alternative
        max_research_rounds=5,
        min_confidence=7,
    )

    print("\n" + "=" * 80 + "\nFINAL ANSWER\n" + "=" * 80)
    print(answer)
