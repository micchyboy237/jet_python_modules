import shutil
from pathlib import Path

from jet.file.utils import save_file
from jet.libs.unstructured_lib.jet_examples.vector_rag.rag_llm import (
    LlamaCppLLM,
)
from rich.console import Console

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

console = Console()
llm = LlamaCppLLM()

messages = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Explain what embeddings are."},
]

console.rule("LLM Example", style="bold blue")
console.print("[cyan]Generating response...[/cyan]")

response = llm.generate(messages, temperature=0.0)

console.print("[green]LLM Response:[/green]")
console.print(response, markup=False)

save_file(
    {
        "messages": messages,
        "response": response,
    },
    OUTPUT_DIR / "llm_response.json",
)
