import logging
import re
import time
from pathlib import Path
from typing import Any

import requests
from jet.adapters.llama_cpp.types import LLAMACPP_LLM_KEYS
from jet.libs.smolagents.custom_models import OpenAIModel
from markdownify import markdownify
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
    WebSearchTool,
    tool,
)

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

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
    model_id: LLAMACPP_LLM_KEYS | None = None,
    logs_dir: str | Path | None = None,
) -> OpenAIModel:
    if model_id is None:
        model_id = "qwen3-instruct-2507:4b"

    return OpenAIModel(
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        logs_dir=logs_dir,
    )


@tool
def visit_webpage(url: str, max_length: int = 4000) -> str:  # ← much safer default
    """
    Fetches the webpage at the given URL, converts it to clean markdown,
    and returns the content (or an error message if fetching fails).

    Args:
        url: The full URL of the webpage to fetch (e.g. "https://example.com").
        max_length: Maximum length of the returned markdown string before truncation.
                    Defaults to 12000 characters.

    Returns:
        Clean markdown representation of the page, or an error string.
    """
    headers_list = [
        {"User-Agent": "Mozilla/5.0 (compatible; DeepResearchBot/1.0)"},
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
    ]

    for attempt in range(1, 5):
        try:
            headers = headers_list[min(attempt - 1, len(headers_list) - 1)]
            response = requests.get(url, headers=headers, timeout=10 + attempt * 2)
            response.raise_for_status()

            md = markdownify(response.text).strip()
            md = re.sub(r"\n{3,}", "\n\n", md)

            if len(md) > max_length:
                md += "\n\n[... truncated ...]"

            return f"[Success after {attempt} attempt(s)]\n{md}"

        except Exception as e:
            time.sleep(1.2**attempt)  # backoff
            if attempt == 4:
                return f"[Failed after 4 attempts] {url} → {str(e)}"

    return "Unreachable code path (should not happen)"


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


def create_web_research_agent(model):
    return ToolCallingAgent(
        tools=[
            WebSearchTool(),
            visit_webpage,
            extract_key_facts,
        ],
        model=model,
        name="web_research_agent",
        description=(
            "A specialized agent that performs web searches, visits pages, "
            "and extracts relevant information. Use this agent when you need "
            "fresh information from the internet or need to dive deeper into "
            "specific sources. Provide clear, specific instructions."
        ),
        max_steps=12,
        verbosity_level=LogLevel.INFO,
        provide_run_summary=True,  # ← important for token saving
    )


def create_evidence_evaluator_agent(model):
    # Slightly better prompt inside description (helps model output parseable text)
    description = (
        "Evaluates collected evidence. Output format:\n"
        "CONFIDENCE: <0-10 integer>\nSUFFICIENT: yes/no\nMISSING: ...\nCONTRADICTIONS: ...\nSOURCES: url1, url2, ...\nREASON: ..."
    )
    return ToolCallingAgent(
        tools=[],
        model=model,
        name="evidence_evaluator",
        description=description,
        max_steps=4,
        verbosity_level=LogLevel.INFO,
        provide_run_summary=True,
    )


def create_query_refiner(model):
    return ToolCallingAgent(
        tools=[],
        model=model,
        name="query_refiner",
        description=(
            "Reformulates the original user query into 1–3 more precise, "
            "effective search queries when initial results are poor, "
            "ambiguous or missing key aspects. "
            "Returns a list of improved queries as bullet points."
        ),
        max_steps=5,
        provide_run_summary=True,
    )


def create_final_formatter(model):
    return ToolCallingAgent(
        tools=[],
        model=model,
        name="final_formatter",
        description=(
            "Takes raw research output, evidence summaries and sources, "
            "then produces clean, well-structured markdown with inline "
            "citations [1], [2]… and a References section at the bottom."
        ),
        max_steps=4,
        provide_run_summary=True,
    )


# ────────────────────────────────────────────────
# Update manager creation


def create_deep_research_manager(model):
    web_agent = create_web_research_agent(model)
    eval_agent = create_evidence_evaluator_agent(model)
    refiner_agent = create_query_refiner(model)
    formatter_agent = create_final_formatter(model)

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
        description=(
            "Top-level research coordinator. Uses specialized sub-agents to "
            "deeply investigate a question, gather evidence, evaluate it, "
            "refine queries when stuck, and produce clean final answers."
        ),
        max_steps=20,
        verbosity_level=LogLevel.INFO,
        additional_authorized_imports=["datetime"],
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
    model = create_local_model(
        temperature=temperature, model_id=model_id, logs_dir=OUTPUT_DIR / "llm_logs"
    )
    manager = create_deep_research_manager(model)

    round_summaries: list[str] = []
    evidence_accumulator: list[dict[str, Any]] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[bold]{task.completed}/{task.total} rounds"),
    )

    def build_evidence_summary() -> str:
        if not evidence_accumulator:
            return "(no evidence collected yet)"

        # Keep only last N rounds + very compact representation
        recent = evidence_accumulator[-history_window:]
        lines = []
        start_idx = len(evidence_accumulator) - len(recent) + 1
        for idx, ev in enumerate(recent, start_idx):
            srcs = ", ".join(ev.get("sources", [])[:3])  # now actually used
            facts = ev.get("raw_output", "")[:180].replace("\n", " ").strip()
            line = f"Round {idx} (conf {ev.get('confidence', '?')}, suff {ev.get('sufficient', '?')}): {facts}"
            if srcs:
                line += f"  [sources: {srcs}]"
            if len(line) > 240:
                line = line[:237] + "..."
            lines.append(line)

        if len(evidence_accumulator) > history_window:
            lines.insert(
                0,
                f"... {len(evidence_accumulator) - history_window} earlier rounds omitted ...",
            )

        return "\n".join(lines) if lines else "(no usable evidence summary)"

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

            # ── Rough token estimation heuristic (very approximate) ──
            estimated_tokens = len(current_plan) // 4 + len(result) // 4 + 1500
            if estimated_tokens > 6500:  # safety margin under 8192
                logger.warning(
                    f"[Round {round_num}] Estimated tokens ≈ {estimated_tokens} → forcing early final format"
                )
                break

            # Simple best-effort parsing of evaluator output
            confidence = None
            sufficient = "no"
            missing = ""
            contradictions = ""
            sources: list[str] = re.findall(r"https?://[^\s,]+", result)

            for line in result.split("\n"):
                line = line.strip()
                if line.startswith("CONFIDENCE:"):
                    try:
                        confidence = int(line.split(":", 1)[1].strip().split("/")[0])
                    except Exception:
                        pass
                elif line.startswith("SUFFICIENT:"):
                    sufficient = "yes" if "yes" in line.lower() else "no"
                elif line.startswith("MISSING:"):
                    missing = line.split(":", 1)[1].strip()
                elif line.startswith("CONTRADICTIONS:"):
                    contradictions = line.split(":", 1)[1].strip()

            # Keep raw_output very short for next iterations
            evidence_accumulator.append(
                {
                    "round": round_num,
                    "confidence": confidence,
                    "sufficient": sufficient,
                    "missing": missing,
                    "contradictions": contradictions,
                    "sources": sources,
                    "raw_output": result[:240].replace("\n", " ").strip() + "...",
                }
            )

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
