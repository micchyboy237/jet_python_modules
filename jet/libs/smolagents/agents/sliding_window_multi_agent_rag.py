import logging
import time
from datetime import timedelta

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from smolagents import CodeAgent, InferenceClientModel, ToolCallingAgent

console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("SlidingWindowRAG")


class SlidingWindowMultiAgentRAG:
    """
    A reusable class for answering a query from a long RAG context using smolagents multi-agent setup.

    - External chunking with overlap for sliding window effect.
    - Dedicated sub-agent updates rolling summary iteratively.
    - Manager agent synthesizes final answer.
    - Rich logging + enhanced progress tracking
    """

    def __init__(
        self,
        model_id: str | None = None,
        chunk_size: int = 5000,  # characters (~ reasonable token limit safety margin)
        overlap: int = 500,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Logging control
        self.verbose = False  # set to True to log agent prompts & responses
        self.summary_preview_chars = 120

        logger.info("[bold cyan]Initializing SlidingWindowMultiAgentRAG[/]")
        logger.debug(f"chunk_size = {chunk_size:,} chars | overlap = {overlap:,} chars")

        # Import locally to avoid circular imports
        from jet.libs.smolagents.utils.model_utils import create_local_model

        # Model (free HF inference by default; override with model_id for specific model)
        model = (
            InferenceClientModel(model_id=model_id)
            if model_id
            else create_local_model(agent_name="manager_agent")
        )

        # Sub-agent specialized for summary updates (safe, no tools/code execution)
        self.analyzer = ToolCallingAgent(
            tools=[],
            model=model,
            name="SummaryUpdater",
            description="Updates a rolling summary by incorporating relevant info from a new text chunk given the query and previous summary. Returns only the updated summary.",
        )

        # Manager agent that can delegate to the analyzer if needed
        self.manager = CodeAgent(
            tools=[],
            model=model,
            managed_agents=[self.analyzer],
        )

    def _get_chunks(self, text: str):
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.overlap
            if end >= len(text):
                break
        return chunks

    def run(self, query: str, text: str) -> str:
        """Process the long context and return the answer."""
        # Direct handling for short text
        if len(text) <= self.chunk_size:
            task = f"""Query: {query}

Text: {text}

Answer the query based on the provided text."""
            logger.info("[bold cyan]Short document → direct processing[/]")
            with console.status(
                "[bold magenta]Running manager agent...", spinner="dots"
            ):
                return self.manager.run(task)

        # ── Long document ────────────────────────────────────────
        chunks = self._get_chunks(text)
        logger.info(
            f"[bold cyan]Long document → sliding window[/] • [dim]{len(chunks)} chunks[/dim]"
        )

        summary = ""
        chunk_times = []

        # Better progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                f"[cyan]Processing chunks • summary len = {len(summary):,}",
                total=len(chunks),
            )

            start_total = time.perf_counter()

            for idx, chunk in enumerate(chunks):
                chunk_start = time.perf_counter()

                task_prompt = f"""Query: {query}

Previous summary:
{summary if summary else "None yet."}

New chunk:
{chunk}

Update the summary with any new information relevant to the query. Keep it concise, structured, and focused only on query-relevant details.

Output ONLY the updated summary (no explanations or prefixes)."""

                if self.verbose:
                    logger.debug(
                        f"[Chunk {idx + 1}] Sending prompt ({len(task_prompt):,} chars)"
                    )

                with console.status(
                    f"[dim]Analyzing chunk {idx + 1}/{len(chunks)}...[/dim]",
                    spinner="dots3",
                ):
                    result = self.analyzer.run(task_prompt)

                summary = result.strip()

                chunk_time = time.perf_counter() - chunk_start
                chunk_times.append(chunk_time)

                preview = summary[: self.summary_preview_chars] + (
                    "…" if len(summary) > self.summary_preview_chars else ""
                )

                progress.update(
                    task_id,
                    advance=1,
                    description=f"[cyan]Processing chunks • summary len = {len(summary):,}[/cyan]",
                )

                if (idx + 1) % 5 == 0 or idx == len(chunks) - 1:
                    logger.info(
                        f"[dim]Chunk {idx + 1:2d}/{len(chunks)}[/dim]  •  "
                        f"summary = {len(summary):4,} chars  •  "
                        f"time = {chunk_time:.2f}s  •  "
                        f"preview = [italic dim]{preview}[/italic dim]"
                    )

        total_time = time.perf_counter() - start_total
        avg_chunk_time = sum(chunk_times) / len(chunk_times) if chunk_times else 0

        logger.info(
            f"[bold green]All {len(chunks)} chunks processed[/] in {timedelta(seconds=int(total_time))}"
        )
        logger.info(f"[dim]Avg chunk time: {avg_chunk_time:.2f}s[/dim]")

        final_task = f"""Query: {query}

Complete relevant summary extracted from the entire document:
{summary}

Provide a clear, comprehensive final answer to the query based on this summary.
If no relevant information was found, state that explicitly."""

        logger.info("[bold magenta]Generating final answer...[/bold magenta]")

        with console.status(
            "[bold magenta]Manager agent synthesizing final answer...",
            spinner="bouncingBall",
        ):
            answer = self.manager.run(final_task)

        logger.info("[bold green]Processing complete![/bold green]")

        # Nicer final presentation
        console.print(
            Panel(
                answer.strip(),
                title=f"[bold green]Final Answer to:[/bold green] [i]{query}[/i]",
                border_style="green",
                expand=True,
            )
        )

        return answer


if __name__ == "__main__":
    file_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/smolagents/agents/data/gutenberg_book_shortened.txt"
    with open(file_path, encoding="utf-8") as f:
        long_text = f.read()

    rag = SlidingWindowMultiAgentRAG()
    query = "What is the main conclusion of the document?"
    answer = rag.run(query, long_text)

    print(answer)
