import asyncio
import json
import uuid
from pathlib import Path

from jet.adapters.llama_cpp.llm_utils_observed import (
    achat,
)
from jet.adapters.llama_cpp.rag_utils import RankedChunkResult, find_snapshot
from jet.adapters.llama_cpp.token_utils import count_tokens
from jet.libs.llama_cpp.usage.chat_stream_types import StreamCompletionResult
from jet.scrapers.browser.agent.memory import AgentMemory
from jet.scrapers.browser.agent.registry import PlaywrightToolRegistry
from jet.scrapers.browser.agent.templates import (
    SYSTEM_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE,
)
from jet.scrapers.browser.agent.tools import AgentDecision
from jet.scrapers.browser.agent.utils import get_step_dir, save_json, save_text
from jet_telemetry import agent, chain, llm, performance_monitor, tool
from rich.console import Console

console = Console()


class LocalAgent:
    def __init__(
        self,
        api_base: str,
        model: str,
        registry: PlaywrightToolRegistry,
        output_dir: Path,
        max_message_tokens: int = 12000,
        max_snapshot_tokens: int = 3000,
        max_retries_per_step: int = 3,
    ):
        self.model = model
        self.registry = registry
        self.output_dir = output_dir
        self.max_message_tokens = max_message_tokens
        self.max_snapshot_tokens = max_snapshot_tokens
        self.max_retries_per_step = max_retries_per_step
        self._count_tokens_fn = lambda text: count_tokens(
            text, model=self.model, use_server=False
        )
        self.prev_url = "Unknown"
        self.total_input_tokens = 0
        self.step_token_counts = []
        self.session_id = str(uuid.uuid4())[:8]
        self.history = []
        self.memory = AgentMemory()

    @chain(name="refresh-browser-snapshot")
    async def _refresh_snapshot(self, step_dir: Path) -> str:
        """Refreshes the browser snapshot if the page is still open."""
        if self.registry.toolbox.is_closed:
            return ""
        try:
            await asyncio.sleep(1)
            return await self.registry.toolbox.browser_snapshot(step_dir=step_dir)
        except Exception as e:
            console.print(f"[yellow]Failed to refresh snapshot: {e}[/yellow]")
            return ""

    @chain(name="generate-retrieval-query")
    async def _generate_query(
        self,
        goal: str,
        current_snapshot: str,
        last_result: str,
        memory_context: str = "",
    ) -> str:
        """
        Generates a concise query for RAG retrieval based on the current goal,
        page snapshot, last action result, AND accumulated memory.
        """
        system_prompt = (
            "You are an assistant helping a browser agent retrieve relevant context from a page snapshot. "
            "Generate a short, specific search query that will help find elements or text relevant to the user's goal. "
            "Focus on interactive elements, labels, or content mentioned in the goal. "
            "Do not include explanations, just the query string."
        )

        # Include memory context in the prompt to help generate better queries
        memory_hint = (
            f"\nRelevant History: {memory_context[:500]}" if memory_context else ""
        )

        user_prompt = (
            f"User Goal: {goal}{memory_hint}\n"
            f"Last Action Result: {last_result}\n"
            f"Current Page Snapshot (truncated if long): {current_snapshot[:2000]}...\n"
            "Query:"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            result: StreamCompletionResult = await achat(
                prompt_or_messages=messages,
                model=self.model,
                temperature=0.1,
                max_tokens=50,
            )
            query = result.content.strip()
            if query.startswith('"') and query.endswith('"'):
                query = query[1:-1]
            return query
        except Exception as e:
            console.print(
                f"[yellow]Failed to generate query, falling back to goal: {e}[/yellow]"
            )
            return goal

    @chain(name="extract-snapshot-context")
    async def _ensure_snapshots_token_limit(
        self, query: str, current_snapshot: str, memory_context: str = ""
    ) -> tuple[str, list[RankedChunkResult], int]:
        """Extracts relevant context using RAG and ensures it fits within snapshot token limits.
        Now includes memory context in the searchable text."""

        # Combine current snapshot with memory for RAG chunking/searching
        text_to_chunk = current_snapshot
        if memory_context:
            text_to_chunk += "\n\n" + memory_context

        rag_context, ranked_results, _ = find_snapshot(
            query=query,
            text=text_to_chunk,  # Pass combined text
            top_k=10,
            min_score=3.0,
            model=self.model,
            criteria="Focus on interactive elements, text content, and layout structure relevant to the query.",
            include_reasoning=True,
            max_tokens=self.max_snapshot_tokens,
            chunk_size=500,
            chunk_overlap=100,
            embed_chunk_size=300,
            embed_chunk_overlap=100,
            show_progress=True,
            min_high_results=3,
            min_results=5,
            candidate_k=10,
        )
        rag_context_tokens = count_tokens(rag_context)
        return rag_context, ranked_results, rag_context_tokens

    def _ensure_message_token_limit(
        self,
        system_prompt: str,
        messages: list[dict],
        tool_registry: list[dict],
    ) -> tuple[list[dict], int]:
        """
        Ensures total message tokens do not exceed max_message_tokens.
        Truncates oldest history steps if necessary.
        """
        step_tokens = count_tokens(
            content=messages,
            model=self.model,
            tools=tool_registry,
            use_server=False,
        )
        while step_tokens > self.max_message_tokens and len(messages) > 1:
            if len(messages) >= 3:
                removed_user = messages.pop(1)
                removed_assistant = messages.pop(1)
                console.print(
                    f"[dim]Truncated history to fit token limit. Removed step.[/dim]"
                )
            else:
                break
            step_tokens = count_tokens(
                content=messages,
                model=self.model,
                tools=tool_registry,
                use_server=False,
            )
        if step_tokens > self.max_message_tokens:
            raise ValueError(
                f"LLM input messages ({step_tokens} tokens) still exceed max allowed ({self.max_message_tokens}) even after truncation."
            )
        return messages, step_tokens

    @chain(name="prepare-llm-input")
    def _prepare_input(
        self,
        step_index: int,
        step_dir: Path,
        system_prompt: str,
        user_prompt_content: str,
        messages: list,
        context_for_llm: str,
        step_tokens: int,
        snapshot_context_tokens: int,
        ranked_results: list,
    ):
        """Saves input data and metrics for the current step."""
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
            context_for_llm,
        )
        save_json(
            step_dir / "token_metrics.json",
            {
                "step": step_index,
                "input_tokens": step_tokens,
                "cumulative_tokens": self.total_input_tokens,
                "snapshot_context_tokens": snapshot_context_tokens,
                "ranked_chunks_count": len(ranked_results),
            },
        )

    @chain(name="capture-output-state")
    async def _capture_output_state(self, step_index: int, step_dir: Path):
        """Captures the browser state after a tool execution that changes the DOM."""
        if self.registry.toolbox.is_closed:
            return
        console.print("[dim]State changed. Capturing output state...[/dim]")
        try:
            output_state = await self.registry.toolbox.browser_snapshot(
                step_dir=step_dir
            )
            output_link = save_text(step_dir / "output_snapshot.md", output_state)
            console.print(f"[dim]Output snapshot saved: {output_link}[/dim]")
            await self.registry.toolbox.browser_take_screenshot(
                filename=f"step_{step_index}_end.png", step_dir=step_dir
            )
        except Exception as e:
            console.print(f"[yellow]Failed to capture end state: {e}[/yellow]")

    @tool(name="execute-agent-tool")
    async def _execute_tool_with_tracking(
        self, decision: AgentDecision, step_dir: Path
    ) -> str:
        """Executes the tool and returns the result string for the next iteration."""
        console.print(f"\n[dim]Thought: {decision.thought}[/dim]")
        console.print(f"[bold yellow]Executing:[/bold yellow] {decision.tool}")
        try:
            last_result = await self._execute_tool(decision, step_dir)
            return last_result
        except Exception as e:
            console.print(f"[bold red]Error in tool execution:[/bold red] {e}")
            return f"Error: {str(e)}"

    @agent(name="browser-agent-loop", session_id=None)
    async def think_and_act(
        self, goal: str, initial_snapshot: str, max_steps: int = 15, **kwargs
    ) -> str:
        tool_registry = self.registry.get_tool_schema()
        tool_descriptions = json.dumps(tool_registry, indent=2)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=tool_descriptions
        )
        self.history = []
        self.total_input_tokens = 0
        self.step_token_counts = []
        self.memory.reset()  # Reset memory for new session

        last_result = "None"
        current_snapshot = initial_snapshot
        self.prev_url = await self.registry.toolbox._get_current_url()
        response_format = AgentDecision

        for i in range(max_steps):
            step_index = i + 1
            step_dir = get_step_dir(step_index, self.output_dir)
            success = False

            for attempt in range(1, self.max_retries_per_step + 1):
                try:
                    if attempt > 1:
                        console.print(
                            f"[bold yellow]⚠️ Retrying Step {step_index} (Attempt {attempt}/{self.max_retries_per_step})...[/bold yellow]"
                        )
                        await asyncio.sleep(2)
                        refreshed = await self._refresh_snapshot(step_dir)
                        if refreshed:
                            current_snapshot = refreshed
                        else:
                            raise Exception("Browser closed during retry wait.")

                    console.print(
                        f"[bold cyan]Step {step_index} (Attempt {attempt}):[/bold cyan] Thinking..."
                    )

                    if i > 0 and attempt == 1:
                        refreshed = await self._refresh_snapshot(step_dir)
                        if refreshed:
                            current_snapshot = refreshed

                    current_url = "Unknown"
                    if (
                        self.registry.toolbox.page
                        and not self.registry.toolbox.page.is_closed()
                    ):
                        try:
                            current_url = self.registry.toolbox.page.url
                        except:
                            pass

                    # Get Memory Context
                    memory_context = self.memory.get_context_string()

                    query = await self._generate_query(
                        goal, current_snapshot, last_result, memory_context
                    )
                    console.print(f"[dim]Generated RAG Query: {query[:100]}...[/dim]")

                    (
                        context_for_llm,
                        ranked_results,
                        snapshot_context_tokens,
                    ) = await self._ensure_snapshots_token_limit(
                        query,
                        current_snapshot,
                        memory_context,  # Pass memory here
                    )

                    input_link = save_text(
                        step_dir / "input_snapshot.md", context_for_llm
                    )
                    console.print(f"[dim]Input context saved: {input_link}[/dim]")
                    console.print(
                        f"[dim]Snapshot Context Tokens: {snapshot_context_tokens}/{self.max_snapshot_tokens}[/dim]"
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
                        snapshot=context_for_llm,
                        last_result=last_result,
                    )

                    self.history.append(
                        {"role": "user", "content": user_prompt_content}
                    )
                    messages = [
                        {"role": "system", "content": system_prompt}
                    ] + self.history

                    messages, step_tokens = self._ensure_message_token_limit(
                        system_prompt=system_prompt,
                        messages=messages,
                        tool_registry=tool_registry,
                    )

                    self._prepare_input(
                        step_index,
                        step_dir,
                        system_prompt,
                        user_prompt_content,
                        messages,
                        context_for_llm,
                        step_tokens,
                        snapshot_context_tokens,
                        ranked_results,
                    )

                    decision = await self._call_llm(
                        messages, response_format, step_dir, **kwargs
                    )

                    assistant_msg_content = json.dumps(decision.model_dump())
                    self.history.append(
                        {"role": "assistant", "content": assistant_msg_content}
                    )

                    if decision.tool == "final_answer":
                        decision.finished = True

                    if decision.finished:
                        console.print("[bold green]Goal achieved.[/bold green]")
                        self._save_token_summary(step_index)
                        return "Goal achieved."

                    last_result = await self._execute_tool_with_tracking(
                        decision, step_dir
                    )

                    # Update Memory after successful tool execution
                    self.memory.add_step(step_index, decision, last_result, current_url)

                    self.prev_url = current_url

                    if not self.registry.toolbox.is_closed:
                        if self.registry.requires_state_refresh(decision.tool):
                            await self._capture_output_state(step_index, step_dir)
                        else:
                            console.print(
                                "[dim]Tool passive/terminal. Skipping output snapshot refresh.[/dim]"
                            )

                    if self.registry.toolbox.is_closed:
                        self._save_token_summary(step_index)
                        return "Browser closed."

                    success = True
                    break

                except Exception as e:
                    console.print(
                        f"[bold red]Error in step {step_index} (Attempt {attempt}):[/bold red] {e}"
                    )
                    last_result = f"Error: {str(e)}"
                    save_text(step_dir / f"error_attempt_{attempt}.log", str(e))
                    if attempt == self.max_retries_per_step:
                        raise e

            if not success:
                console.print(
                    "[bold red]Step failed after all retries. Stopping.[/bold red]"
                )
                return "Step failed after retries."

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

    @llm(model_name=None, provider="llama_cpp")
    @performance_monitor(threshold_ms=5000)
    async def _call_llm(self, messages, response_format, step_dir, **kwargs):
        """
        Calls the LLM using the observed utility which handles
        streaming, structured output parsing, and telemetry.
        """
        model_name = kwargs.get("model", self.model)
        extra_body_params = {
            "top_k": 40,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        }
        result: StreamCompletionResult = await achat(
            prompt_or_messages=messages,
            model=model_name,
            temperature=0.3,
            top_p=0.95,
            presence_penalty=1.5,
            frequency_penalty=0.0,
            max_tokens=16384,
            response_format=response_format,
            extra_body_params=extra_body_params,
            seed=42,
        )
        save_text(step_dir / "llm_raw_response.txt", result.content)
        if result.structured and result.structured.success:
            decision = result.structured.parsed
            save_json(step_dir / "llm_decision.json", decision.model_dump())
            return decision
        else:
            error_msg = (
                result.structured.error if result.structured else "Unknown error"
            )
            console.print(
                f"[yellow]Structured parsing failed: {error_msg}. Falling back to manual parse.[/yellow]"
            )
            cleaned_content = self._clean_json_content(result.content)
            try:
                decision = AgentDecision.model_validate_json(cleaned_content)
                save_json(step_dir / "llm_decision.json", decision.model_dump())
                return decision
            except Exception as e:
                console.print(
                    f"[bold red]Failed to parse LLM decision manually:[/bold red] {e}"
                )
                console.print(f"Raw content: {result.content[:200]}...")
                raise e

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
