import ast
import logging
import time
from datetime import timedelta

from jet.libs.smolagents.utils.model_utils import create_local_model
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
logger = logging.getLogger("CodeSummaryMultiAgent")


class CodeStructureExtractor:
    """
    Extracts structural elements from Python code using AST.
    Falls back gracefully if parsing fails.
    """

    @staticmethod
    def extract(code: str) -> dict[str, list[str]]:
        try:
            tree = ast.parse(code)
        except Exception:
            return {"raw": [code]}

        imports = []
        definitions = []

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.unparse(node))
            elif isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                definitions.append(ast.unparse(node))

        return {
            "imports": imports,
            "definitions": definitions,
        }


class CodeSummaryMultiAgent:
    """
    Multi-agent system specialized for summarizing code
    WITHOUT losing structural or API information.

    Pipeline:
        1. Extract structure (AST-aware)
        2. Summarize implementation per unit
        3. Verify invariants preserved
        4. Synthesize final structured summary
    """

    def __init__(
        self,
        model_id: str | None = None,
        chunk_size: int = 8000,
    ):
        self.chunk_size = chunk_size
        self.verbose = False

        logger.info("[bold cyan]Initializing CodeSummaryMultiAgent[/]")

        model = (
            InferenceClientModel(model_id=model_id)
            if model_id
            else create_local_model(agent_name="code_summary_manager")
        )

        # ─────────────────────────────────────────────
        # STRUCTURE AGENT
        # ─────────────────────────────────────────────
        self.structure_agent = ToolCallingAgent(
            tools=[],
            model=model,
            name="StructureAgent",
            description=(
                "Extracts and rewrites structural components of code while "
                "preserving signatures, decorators, inheritance, types, "
                "exceptions, and public APIs."
            ),
        )

        # ─────────────────────────────────────────────
        # SUMMARIZER AGENT
        # ─────────────────────────────────────────────
        self.summarizer_agent = ToolCallingAgent(
            tools=[],
            model=model,
            name="ImplementationSummarizer",
            description=(
                "Condenses implementation details into precise, lossless "
                "semantic descriptions while preserving logic and side effects."
            ),
        )

        # ─────────────────────────────────────────────
        # VERIFIER AGENT
        # ─────────────────────────────────────────────
        self.verifier_agent = ToolCallingAgent(
            tools=[],
            model=model,
            name="InvariantVerifier",
            description=(
                "Ensures no structural information was lost. "
                "Verifies presence of all functions, classes, signatures, "
                "imports, and contracts."
            ),
        )

        # ─────────────────────────────────────────────
        # MANAGER AGENT
        # ─────────────────────────────────────────────
        self.manager = CodeAgent(
            tools=[],
            model=model,
            managed_agents=[
                self.structure_agent,
                self.summarizer_agent,
                self.verifier_agent,
            ],
        )

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def summarize(self, code: str) -> str:
        """
        Produce a structure-preserving, lossless summary of code.
        """

        start_time = time.perf_counter()

        structure = CodeStructureExtractor.extract(code)

        # Fallback: if AST failed
        if "raw" in structure:
            logger.warning("[yellow]AST parsing failed → fallback mode[/yellow]")
            return self._fallback_summary(code)

        imports = "\n".join(structure.get("imports", []))
        definitions = structure.get("definitions", [])

        logger.info(
            f"[bold cyan]Found {len(definitions)} top-level definitions[/bold cyan]"
        )

        summarized_units = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                "[cyan]Summarizing definitions...[/cyan]",
                total=len(definitions),
            )

            for definition in definitions:
                prompt = f"""
You are summarizing code WITHOUT losing information.

Rules:
- Preserve function/class signature EXACTLY.
- Preserve decorators.
- Preserve inheritance.
- Preserve type hints.
- Preserve raised exceptions.
- Preserve side effects.
- Replace implementation body with structured semantic summary.
- Do NOT remove any public API elements.

Code:
{definition}

Output rewritten version.
"""

                result = self.summarizer_agent.run(prompt)
                summarized_units.append(result.strip())

                progress.update(task_id, advance=1)

        combined_summary = f"""
# IMPORTS
{imports}

# DEFINITIONS
{chr(10).join(summarized_units)}
"""

        # ─────────────────────────────────────────────
        # Verification step
        # ─────────────────────────────────────────────
        verification_prompt = f"""
Original code:
{code}

Generated summary:
{combined_summary}

Verify:
- All functions preserved?
- All classes preserved?
- All signatures identical?
- All decorators preserved?
- All inheritance preserved?
- All imports preserved?

If something is missing, explicitly list it.
Otherwise respond ONLY with: VERIFIED
"""

        verification_result = self.verifier_agent.run(verification_prompt)

        total_time = time.perf_counter() - start_time
        logger.info(
            f"[bold green]Completed in {timedelta(seconds=int(total_time))}[/bold green]"
        )

        console.print(
            Panel(
                combined_summary.strip(),
                title="[bold green]Code Summary[/bold green]",
                border_style="green",
            )
        )

        if "VERIFIED" not in verification_result:
            logger.warning("[red]Verification flagged potential information loss[/red]")
            logger.warning(verification_result)

        return combined_summary.strip()

    # ─────────────────────────────────────────────
    # Fallback (non-AST languages)
    # ─────────────────────────────────────────────

    def _fallback_summary(self, code: str) -> str:
        """
        Sliding window summarization when AST fails
        (for non-Python languages or malformed code).
        """

        chunks = [
            code[i : i + self.chunk_size] for i in range(0, len(code), self.chunk_size)
        ]

        summary = ""

        for chunk in chunks:
            prompt = f"""
Summarize this code chunk without losing information.
Preserve signatures, APIs, and contracts.

Previous summary:
{summary}

New chunk:
{chunk}

Return updated summary only.
"""
            summary = self.summarizer_agent.run(prompt).strip()

        return summary
