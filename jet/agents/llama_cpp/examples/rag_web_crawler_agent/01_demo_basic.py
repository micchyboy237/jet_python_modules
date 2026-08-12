import asyncio
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for file-only rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from jet.agents.llama_cpp.rag_web_crawler_agent import run_webswarm
from matplotlib.patches import Patch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ─── Configuration ────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
QUERY = "What are the deployment options for LangGraph Platform?"
ROOT_URL = "https://langchain-ai.github.io/langgraph/"


# ─── Dual Output Writer ───────────────────────────────────────────────────────
class TeeWriter:
    """Writes to both terminal (via Rich) and a log file simultaneously."""

    def __init__(self, term_console: Console, log_file: TextIO):
        self._term = term_console
        self._log = log_file

    def write(self, text: str) -> int:
        self._log.write(text)
        self._log.flush()
        if text.strip():
            self._term.print(text, end="", highlight=False)
        elif text:
            self._term.file.write(text)
            self._term.file.flush()
        return len(text)

    def flush(self):
        self._log.flush()
        self._term.file.flush()


# ─── Chart Generators ─────────────────────────────────────────────────────────
def generate_score_chart(result: dict, output_dir: Path) -> Path | None:
    """Generate relevance score progression chart across iterations."""
    retrieve_steps = result.get("retrieve_steps", [])
    if not retrieve_steps:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    iterations = []
    all_scores = []
    max_scores = []
    mean_scores = []
    chunks_added = []

    for step in retrieve_steps:
        it = step["iteration"]
        scores = [c["score"] for c in step.get("reranked_chunks", [])]
        iterations.append(it)
        all_scores.extend([(it, s) for s in scores])
        max_scores.append(max(scores) if scores else 0.0)
        mean_scores.append(sum(scores) / len(scores) if scores else 0.0)
        chunks_added.append(step.get("chunks_added_to_kb", 0))

    # Scatter: individual chunk scores
    if all_scores:
        xs, ys = zip(*all_scores)
        ax.scatter(
            xs, ys, alpha=0.5, s=40, color="#4A90D9", label="Chunk Score", zorder=3
        )

    # Line: max score per iteration
    ax.plot(
        iterations,
        max_scores,
        "o-",
        color="#2ECC71",
        linewidth=2,
        markersize=8,
        label="Max Score",
        zorder=4,
    )

    # Line: mean score per iteration
    ax.plot(
        iterations,
        mean_scores,
        "s--",
        color="#F39C12",
        linewidth=1.5,
        markersize=6,
        label="Mean Score",
        zorder=4,
    )

    # Threshold line
    threshold = result.get("config", {}).get("relevance_threshold", 0.15)
    ax.axhline(
        y=threshold,
        color="#E74C3C",
        linestyle=":",
        linewidth=1.5,
        label=f"Threshold ({threshold})",
        zorder=2,
    )

    # Bar: chunks added to KB (secondary axis)
    ax2 = ax.twinx()
    ax2.bar(
        iterations,
        chunks_added,
        alpha=0.15,
        color="#8E44AD",
        width=0.6,
        label="Chunks Added to KB",
        zorder=1,
    )
    ax2.set_ylabel("Chunks Added to KB", color="#8E44AD", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#8E44AD")
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Formatting
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Relevance Score (Sigmoid Normalized)", fontsize=12)
    ax.set_title(
        "RAG Web Crawler: Relevance Score Progression", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(iterations)
    ax.set_xlim(min(iterations) - 0.5, max(iterations) + 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, zorder=0)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

    plt.tight_layout()
    chart_path = output_dir / "score_progression.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return chart_path


def generate_workflow_chart(result: dict, output_dir: Path) -> Path | None:
    """Generate node execution timeline chart showing workflow progression."""
    retrieve_steps = result.get("retrieve_steps", [])
    evaluate_steps = result.get("evaluate_steps", [])
    route_decisions = result.get("route_decisions", [])
    synth_step = result.get("synthesize_step")

    if not retrieve_steps:
        return None

    fig, ax = plt.subplots(figsize=(14, max(4, len(retrieve_steps) * 1.2 + 2)))

    colors = {
        "retrieve": "#3498DB",
        "evaluate": "#E67E22",
        "synthesize": "#2ECC71",
        "route": "#9B59B6",
    }

    y_positions = []
    y_labels = []
    bar_data = []  # (y, start, duration, label, color, annotation)

    time_cursor = 0.0

    for i, (ret, eva) in enumerate(zip(retrieve_steps, evaluate_steps)):
        iteration = ret["iteration"]
        y = len(retrieve_steps) - i  # Top-down ordering
        y_positions.append(y)
        y_labels.append(f"Iter {iteration}")

        # Retrieve bar
        r_width = max(0.5, ret.get("fetched_pages", 1) * 0.3 + 0.5)
        kb_added = ret.get("chunks_added_to_kb", 0)
        bar_data.append(
            (y, time_cursor, r_width, "Retrieve", colors["retrieve"], f"+{kb_added} KB")
        )
        time_cursor += r_width + 0.1

        # Evaluate bar
        e_width = 0.6
        eval_label = eva.get("evaluation", "?")[:4]
        bar_data.append(
            (y, time_cursor, e_width, "Evaluate", colors["evaluate"], eval_label)
        )
        time_cursor += e_width + 0.1

        # Route marker
        route_match = next(
            (r for r in route_decisions if r["iteration"] == iteration), None
        )
        if route_match:
            decision = route_match["decision"]
            ax.plot(
                time_cursor,
                y,
                marker="D",
                color=colors["route"],
                markersize=8,
                zorder=5,
            )
            ax.annotate(
                decision[:4].upper(),
                xy=(time_cursor, y),
                fontsize=7,
                ha="center",
                va="bottom",
                color=colors["route"],
                fontweight="bold",
            )
            time_cursor += 0.3

        time_cursor += 0.3  # Gap between iterations

    # Synthesize bar (final)
    if synth_step:
        y_synth = 0
        y_positions.append(y_synth)
        y_labels.append("Final")
        s_width = max(1.0, synth_step.get("kb_entries_used", 1) * 0.2 + 0.5)
        bar_data.append(
            (
                y_synth,
                0.0,
                s_width,
                "Synthesize",
                colors["synthesize"],
                f"{synth_step.get('kb_entries_used', 0)} srcs",
            )
        )

    # Draw bars
    for y, start, width, label, color, annotation in bar_data:
        ax.barh(
            y,
            width,
            left=start,
            height=0.6,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        text_x = start + width / 2
        if width < 0.8:
            text_x = start + width + 0.05
            ha = "left"
        else:
            ha = "center"
        ax.text(
            text_x,
            y,
            label,
            va="center",
            ha=ha,
            fontsize=8,
            fontweight="bold",
            color="white" if width >= 0.8 else color,
            zorder=4,
        )
        if annotation:
            ann_y = y - 0.22
            ax.text(
                text_x,
                ann_y,
                annotation,
                va="center",
                ha=ha,
                fontsize=7,
                color="#555555",
                style="italic",
                zorder=4,
            )

    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel("Execution Progress →", fontsize=12)
    ax.set_title(
        "RAG Web Crawler: Workflow Node Timeline", fontsize=14, fontweight="bold"
    )
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.2, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend
    legend_elements = [
        Patch(facecolor=colors["retrieve"], label="Retrieve"),
        Patch(facecolor=colors["evaluate"], label="Evaluate"),
        Patch(facecolor=colors["synthesize"], label="Synthesize"),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=colors["route"],
            markersize=8,
            label="Route Decision",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, ncol=4)

    plt.tight_layout()
    chart_path = output_dir / "workflow_timeline.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return chart_path


# ─── Artifact Persistence ─────────────────────────────────────────────────────
def save_artifacts(result: dict, output_dir: Path) -> list[Path]:
    """Persist all run artifacts including node-level I/O traces and charts."""
    saved_files: list[Path] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    # 1. Answer (Markdown)
    answer_path = output_dir / "answer.md"
    answer_content = (
        f"# RAG Web Crawler Result\n\n"
        f"**Query:** {QUERY}\n"
        f"**Root:** {ROOT_URL}\n"
        f"**Generated:** {timestamp}\n\n---\n\n"
        f"{result.get('answer', 'No answer generated.')}\n"
    )
    answer_path.write_text(answer_content, encoding="utf-8")
    saved_files.append(answer_path)

    # 2. Knowledge Base (Full JSON)
    kb_path = output_dir / "knowledge_base.json"
    kb_entries = result.get("knowledge_base", [])
    kb_data = [
        {
            "url": e.get("url", ""),
            "score": round(e.get("score", 0.0), 4),
            "raw_score": round(e.get("raw_score", 0.0), 4),
            "original_chars": e.get("original_chars", 0),
            "truncated_chars": e.get("truncated_chars", 0),
            "content": e.get("content", ""),
        }
        for e in kb_entries
    ]
    kb_path.write_text(
        json.dumps(kb_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    saved_files.append(kb_path)

    # 3. Run Metadata + Config
    meta_path = output_dir / "run_metadata.json"
    metadata = {
        "query": QUERY,
        "root_url": ROOT_URL,
        "iterations": result.get("iterations", 0),
        "pages_visited": result.get("pages_visited", 0),
        "kb_size": len(kb_entries),
        "timestamp": timestamp,
        "output_dir": str(output_dir.resolve()),
        "config": result.get("config", {}),
    }
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    saved_files.append(meta_path)

    # 4. Sources Summary
    sources_path = output_dir / "sources.txt"
    sources = result.get("sources", [])
    sources_lines = [f"[{i + 1}] {url}" for i, url in enumerate(sources)]
    sources_path.write_text(
        "\n".join(sources_lines) or "No sources retrieved.", encoding="utf-8"
    )
    saved_files.append(sources_path)

    # 5. Node-Level Trace: Retrieve Steps
    retrieve_path = output_dir / "trace_retrieve.json"
    retrieve_path.write_text(
        json.dumps(result.get("retrieve_steps", []), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    saved_files.append(retrieve_path)

    # 6. Node-Level Trace: Evaluate Steps
    evaluate_path = output_dir / "trace_evaluate.json"
    evaluate_path.write_text(
        json.dumps(result.get("evaluate_steps", []), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    saved_files.append(evaluate_path)

    # 7. Node-Level Trace: Synthesize Step
    synth_path = output_dir / "trace_synthesize.json"
    synth_step = result.get("synthesize_step")
    synth_path.write_text(
        json.dumps(synth_step, indent=2, ensure_ascii=False) if synth_step else "{}",
        encoding="utf-8",
    )
    saved_files.append(synth_path)

    # 8. Node-Level Trace: Route Decisions
    route_path = output_dir / "trace_routes.json"
    route_path.write_text(
        json.dumps(result.get("route_decisions", []), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    saved_files.append(route_path)

    # 9. Score Progression Chart
    score_chart_path = generate_score_chart(result, output_dir)
    if score_chart_path:
        saved_files.append(score_chart_path)

    # 10. Workflow Timeline Chart
    workflow_chart_path = generate_workflow_chart(result, output_dir)
    if workflow_chart_path:
        saved_files.append(workflow_chart_path)

    # 11. Workflow Graph Diagram
    graph_bytes = result.get("graph_png")
    if graph_bytes:
        graph_path = output_dir / "workflow_graph.png"
        graph_path.write_bytes(graph_bytes)
        saved_files.append(graph_path)

    # 12. Log file
    log_path = output_dir / "agent_trace.log"
    saved_files.append(log_path)

    return saved_files


def display_resource_links(files: list[Path], base_dir: Path):
    """Display saved files as clickable resource links with base names."""
    table = Table(
        title="📁 Saved Artifacts", show_header=True, header_style="bold cyan"
    )
    table.add_column("File", style="green", no_wrap=True)
    table.add_column("Path", style="dim")
    table.add_column("Size", justify="right", style="yellow")

    for f in sorted(files):
        size = f.stat().st_size
        size_str = f"{size:,} B" if size < 1024 else f"{size / 1024:.1f} KB"
        rel_path = f.relative_to(base_dir)
        abs_uri = f"file://{f.resolve()}"
        table.add_row(f"[link={abs_uri}]{f.name}[/link]", str(rel_path), size_str)

    Console().print()
    Console().print(table)
    Console().print()


# ─── Main Entry Point ─────────────────────────────────────────────────────────
async def main():
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_path = OUTPUT_DIR / "agent_trace.log"
    log_file = log_path.open("w", encoding="utf-8")
    term_console = Console(stderr=True)

    tee = TeeWriter(term_console, log_file)
    original_stdout = sys.stdout

    try:
        sys.stdout = tee  # type: ignore[assignment]

        term_console.print(
            Panel(
                f"[bold]Query:[/bold] {QUERY}\n[bold]Root:[/bold] {ROOT_URL}",
                title="🕷️ RAG Web Crawler Agent",
                border_style="blue",
            )
        )

        result = await run_webswarm(query=QUERY, root_url=ROOT_URL)

    except Exception as e:
        term_console.print(f"\n[bold red]❌ Agent failed:[/bold red] {e}")
        raise
    finally:
        sys.stdout = original_stdout
        log_file.close()

    saved_files = save_artifacts(result, OUTPUT_DIR)

    term_console.print(
        Panel(
            f"[bold green]✓ Completed[/bold green] in {result.get('iterations', '?')} iterations | "
            f"{result.get('pages_visited', '?')} pages visited | "
            f"{len(result.get('sources', []))} sources",
            border_style="green",
        )
    )

    display_resource_links(saved_files, OUTPUT_DIR)


if __name__ == "__main__":
    asyncio.run(main())
