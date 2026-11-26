from typing import Optional

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.llm_utils import chat
from rich.console import Console
from turftopic.analyzers.base import Analyzer

console = Console()


class LlamaCppAnalyzer(Analyzer):
    """Analyze topic model with a local Llama.cpp LLM server."""

    def __init__(
        self,
        model_name: str = LLM_MODEL,
        context: Optional[str] = None,
        use_summaries: bool = False,
        system_prompt: Optional[str] = None,
        summary_prompt: Optional[str] = None,
        namer_prompt: Optional[str] = None,
        description_prompt: Optional[str] = None,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt or self.system_prompt
        self.summary_prompt = summary_prompt or self.summary_prompt
        self.namer_prompt = namer_prompt or self.namer_prompt
        self.description_prompt = description_prompt or self.description_prompt
        self.use_summaries = use_summaries
        self.context = context
        console.log(
            f"[dim]Initialized LlamaCppAnalyzer with model: {self.model_name}[/dim]"
        )

    def generate_text(self, prompt: str) -> str:
        """Generates response to a given prompt using local Llama.cpp server."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        console.log(
            f"[dim]Generating text for prompt length: {len(prompt)} chars[/dim]"
        )
        try:
            result = chat(
                prompt_or_messages=messages,
                model=self.model_name,
                max_tokens=512,
                temperature=0.1,
            )
            content = result.content.strip()
            console.log(f"[dim]Generated text length: {len(content)} chars[/dim]")
            return content
        except Exception as e:
            console.print(
                f"[bold red]✗ Error in LlamaCppAnalyzer.generate_text:[/bold red] {e}"
            )
            raise
