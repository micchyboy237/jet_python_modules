from typing import List
from rich.console import Console
from rich.markdown import Markdown

from smolagents import OpenAIModel, ChatMessage


console = Console()

# ────────────────────────────────────────────────────────────────
#  1. Initialize the model pointing to your local llama.cpp server
# ────────────────────────────────────────────────────────────────
model = OpenAIModel(
    model_id="whatever-you-named-it-in-server",     # can be almost anything; llama.cpp usually ignores it
    api_base="http://shawn-pc.local:8080/v1",
    # api_key="lm-studio"                           # optional – many local servers accept dummy key or no auth
    temperature=0.7,
    max_tokens=1024,                                # forwarded to completion call
)


# ────────────────────────────────────────────────────────────────
#  2. Prepare a realistic messages list
# ────────────────────────────────────────────────────────────────
messages: List[ChatMessage] = [
    ChatMessage(
        role="system",
        content="You are a helpful AI coding assistant with a slightly sarcastic personality."
    ),
    ChatMessage(
        role="user",
        content="Write a small Python class that implements a rate limiter using token bucket algorithm."
    ),
]


# ────────────────────────────────────────────────────────────────
#  3. Stream generation + nice real-time printing
# ────────────────────────────────────────────────────────────────
console.print("[bold green]┌─ Thinking …[/bold green]")

full_content = ""
tool_calls_seen = False

for delta in model.generate_stream(
    messages=messages,
    # stop_sequences=["</s>", "[END]"],           # optional – if your model uses them
    # temperature=0.75,                            # you can override here too
):

    # ── usage info sometimes comes in a separate event ───────────
    if delta.token_usage:
        console.print(
            f"\n[dim]→ usage so far: {delta.token_usage}[/dim]",
            style="dim"
        )

    # ── normal text token(s) ─────────────────────────────────────
    if delta.content is not None:
        full_content += delta.content
        console.print(delta.content, end="")   # stream to terminal

    # ── tool call deltas (if model supports tool calling) ────────
    if delta.tool_calls:
        tool_calls_seen = True
        for tc in delta.tool_calls:
            # Usually you would accumulate partial tool calls here
            console.print(f"\n[cyan]→ tool call delta: {tc}[/cyan]")

console.print("\n[bold green]└─ Done ───────────────────────────────────────[/bold green]\n")

# Optional: show final accumulated message in nice markdown
if full_content:
    console.print(Markdown(full_content))
else:
    console.print("[yellow]No content was generated.[/yellow]")

if tool_calls_seen:
    console.print("[yellow]⚠️  Tool calls were requested during streaming[/yellow]")