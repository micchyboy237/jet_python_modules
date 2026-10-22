"""
Agent Core Logic
"""

import json
from pathlib import Path

import openai

# ADDED: Import count_tokens
from jet.adapters.llama_cpp.token_utils import count_tokens
from jet.scrapers.browser.agent.registry import PlaywrightToolRegistry
from jet.scrapers.browser.agent.templates import (
    SYSTEM_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE,
)
from jet.scrapers.browser.agent.tools import AgentDecision, get_decision_schema
from jet.scrapers.browser.agent.utils import get_step_dir, save_json, save_text
from rich.console import Console

console = Console()


class LocalAgent:
    def __init__(
        self,
        api_base: str,
        model: str,
        registry: PlaywrightToolRegistry,
        output_dir: Path,
        max_snapshot_tokens: int = 8000,  # UPDATED: Parameter name and default
    ):
        self.client = openai.AsyncOpenAI(base_url=api_base, api_key="local")
        self.model = model
        self.registry = registry
        self.output_dir = output_dir
        self.max_snapshot_tokens = max_snapshot_tokens  # UPDATED: Store token limit

        # Initialize token counter helper
        self._count_tokens_fn = lambda text: count_tokens(
            text, model=self.model, use_server=False
        )

        self.prev_url = "Unknown"
        # ADDED: Token tracking state
        self.total_input_tokens = 0
        self.step_token_counts = []

    def _truncate_snapshot_by_tokens(self, snapshot: str) -> tuple[str, int]:
        """
        Truncate snapshot to fit within max_snapshot_tokens using binary search.
        Returns truncated snapshot and actual token count.
        """
        if not snapshot:
            return "", 0

        # Quick check if no truncation needed
        total_tokens = self._count_tokens_fn(snapshot)
        if total_tokens <= self.max_snapshot_tokens:
            return snapshot, total_tokens

        # Binary search for the optimal cut point
        left, right = 0, len(snapshot)
        best_truncated = snapshot[: self.max_snapshot_tokens]  # Fallback
        best_token_count = 0

        while left <= right:
            mid = (left + right) // 2
            # Ensure we don't cut in the middle of a unicode char if possible,
            # but for speed we just slice.
            truncated = snapshot[:mid]

            try:
                token_count = self._count_tokens_fn(truncated)
            except Exception:
                token_count = mid  # Rough estimate if tokenizer fails

            if token_count <= self.max_snapshot_tokens:
                best_truncated = truncated
                best_token_count = token_count
                left = mid + 1
            else:
                right = mid - 1

        # Add indication of truncation
        if len(best_truncated) < len(snapshot):
            suffix = "\n\n... [snapshot truncated due to token limit] ..."
            # Check if adding suffix exceeds limit
            if (
                self._count_tokens_fn(best_truncated + suffix)
                <= self.max_snapshot_tokens
            ):
                best_truncated += suffix
                best_token_count = self._count_tokens_fn(best_truncated)

        return best_truncated, best_token_count

    async def think_and_act(
        self, goal: str, initial_snapshot: str, max_steps: int = 15, **kwargs
    ) -> str:
        tool_registry = self.registry.get_tool_schema()
        tool_descriptions = json.dumps(tool_registry, indent=2)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=tool_descriptions
        )

        # Reset tracking for this run
        self.total_input_tokens = 0
        self.step_token_counts = []

        last_result = "None"
        current_state = initial_snapshot
        self.prev_url = await self.registry.toolbox._get_current_url()
        decision_schema = get_decision_schema()
        response_format = {"type": "json_schema", "json_schema": decision_schema}

        for i in range(max_steps):
            step_index = i + 1
            step_dir = get_step_dir(step_index, self.output_dir)
            console.print(f"[bold cyan]Step {step_index}:[/bold cyan] Thinking...")

            if i > 0 and not self.registry.toolbox.is_closed:
                console.print("[dim]Refreshing snapshot before LLM decision...[/dim]")
                try:
                    current_state = await self.registry.toolbox.browser_snapshot(
                        step_dir=step_dir
                    )
                except Exception as e:
                    console.print(f"[yellow]Failed to refresh snapshot: {e}[/yellow]")

            current_url = "Unknown"
            if (
                self.registry.toolbox.page
                and not self.registry.toolbox.page.is_closed()
            ):
                try:
                    current_url = self.registry.toolbox.page.url
                except:
                    pass

            # UPDATED: Use token-based truncation
            truncated_snapshot, snapshot_tokens = self._truncate_snapshot_by_tokens(
                current_state
            )

            input_link = save_text(step_dir / "input_snapshot.md", truncated_snapshot)
            console.print(f"[dim]Input snapshot saved: {input_link}[/dim]")
            console.print(
                f"[dim]Snapshot tokens: {snapshot_tokens}/{self.max_snapshot_tokens}[/dim]"
            )

            try:
                await self.registry.toolbox.browser_take_screenshot(
                    filename=f"step_{step_index}_start.png", step_dir=step_dir
                )
            except Exception as e:
                console.print(
                    f"[yellow]Failed to capture start screenshot: {e}[/yellow]"
                )

            user_prompt_content = USER_PROMPT_TEMPLATE.format(
                goal=goal,
                url=current_url,
                prev_url=self.prev_url,
                snapshot=truncated_snapshot,  # Use truncated version
                last_result=last_result,
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_content},
            ]

            # ADDED: Count tokens for this step
            step_tokens = count_tokens(
                content=messages,
                model=self.model,
                tools=tool_registry,
                use_server=False,
            )

            self.total_input_tokens += step_tokens
            self.step_token_counts.append(step_tokens)

            console.print(
                f"[dim]Step {step_index} input tokens: {step_tokens} (Total: {self.total_input_tokens})[/dim]"
            )

            self._save_step_input(
                step_dir,
                system_prompt,
                user_prompt_content,
                messages,
                truncated_snapshot,
            )

            # Save step token metrics
            save_json(
                step_dir / "token_metrics.json",
                {
                    "step": step_index,
                    "input_tokens": step_tokens,
                    "cumulative_tokens": self.total_input_tokens,
                    "snapshot_tokens": snapshot_tokens,
                },
            )

            try:
                decision = await self._call_llm(
                    messages, response_format, step_dir, **kwargs
                )
                if decision.tool == "final_answer":
                    decision.finished = True
                if decision.finished:
                    console.print("[bold green]Goal achieved.[/bold green]")

                    # ADDED: Save final summary
                    self._save_token_summary(step_index)
                    return "Goal achieved."

                console.print(f"\n[dim]Thought: {decision.thought}[/dim]")
                console.print(f"[bold yellow]Executing:[/bold yellow] {decision.tool}")
                last_result = await self._execute_tool(decision, step_dir)
                self.prev_url = current_url

                if not self.registry.toolbox.is_closed:
                    if self.registry.requires_state_refresh(decision.tool):
                        console.print(
                            "[dim]State changed. Capturing output state...[/dim]"
                        )
                        try:
                            output_state = await self.registry.toolbox.browser_snapshot(
                                step_dir=step_dir
                            )
                            output_link = save_text(
                                step_dir / "output_snapshot.md", output_state
                            )
                            console.print(
                                f"[dim]Output snapshot saved: {output_link}[/dim]"
                            )
                            await self.registry.toolbox.browser_take_screenshot(
                                filename=f"step_{step_index}_end.png", step_dir=step_dir
                            )
                        except Exception as e:
                            console.print(
                                f"[yellow]Failed to capture end state: {e}[/yellow]"
                            )
                    else:
                        console.print(
                            "[dim]Tool passive/terminal. Skipping output snapshot refresh.[/dim]"
                        )

                if self.registry.toolbox.is_closed:
                    self._save_token_summary(step_index)
                    return "Browser closed."

            except Exception as e:
                console.print(f"[bold red]Error in step {step_index}:[/bold red] {e}")
                last_result = f"Error: {str(e)}"
                save_text(step_dir / "error.log", str(e))

        console.print("[bold yellow]Max steps reached.[/bold yellow]")
        self._save_token_summary(max_steps)
        return "Max steps reached."

    def _save_token_summary(self, final_step: int):
        """Save final token usage summary."""
        if not self.step_token_counts:
            return

        summary = {
            "total_steps": final_step,
            "total_input_tokens": self.total_input_tokens,
            "average_tokens_per_step": self.total_input_tokens / final_step
            if final_step > 0
            else 0,
            "model": self.model,
            "max_snapshot_tokens_limit": self.max_snapshot_tokens,
            "step_breakdown": self.step_token_counts,
        }
        save_json(self.output_dir / "token_usage_summary.json", summary)
        console.print(
            f"[bold cyan]Total tokens used: {self.total_input_tokens}[/bold cyan]"
        )

    async def _execute_tool(self, decision, step_dir):
        """Execute tool without managing snapshot state."""
        try:
            result = await self.registry.execute_tool(
                decision.tool, decision.arguments or {}, step_dir=step_dir
            )
            if decision.tool == "final_answer":
                console.print(
                    "[bold green]Final answer provided. Stopping iteration.[/bold green]"
                )
            else:
                console.print(f"[green]Successfully executed {decision.tool}[/green]")
            return result
        except Exception as exec_err:
            console.print(f"[red]Execution error: {exec_err}[/red]")
            return f"Error executing {decision.tool}: {str(exec_err)}"

    async def _call_llm(self, messages, response_format, step_dir, **kwargs):
        create_kwargs = dict(
            model=self.model,
            messages=messages,
            seed=42,
            temperature=0.3,
            top_p=0.95,
            presence_penalty=1.5,
            frequency_penalty=0.0,
            stream_options={"include_usage": True},
            extra_body={
                "top_k": 40,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
            },
            stream=True,
            response_format=response_format,
            **kwargs,
        )
        stream = await self.client.chat.completions.create(**create_kwargs)
        content = ""
        async for part in stream:
            if part.choices and part.choices[0].delta.content:
                content += part.choices[0].delta.content
                print(part.choices[0].delta.content, end="", flush=True)
        content = self._clean_json_content(content)
        save_text(step_dir / "llm_raw_response.txt", content)
        decision = AgentDecision.model_validate_json(content)
        save_json(step_dir / "llm_decision.json", decision.model_dump())
        return decision

    def _save_step_input(
        self, step_dir, system_prompt, user_prompt, messages, snapshot
    ):
        input_data = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "messages": messages,
        }
        save_json(step_dir / "llm_input.json", input_data)

    def _clean_json_content(self, content: str) -> str:
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()
