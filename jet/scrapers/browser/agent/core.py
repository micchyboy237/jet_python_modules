"""
Agent Core Logic
Handles the LLM interaction loop, decision parsing, and tool execution orchestration.
"""

import json
from pathlib import Path

import openai
from jet.scrapers.browser.agent.registry import PlaywrightToolRegistry
from jet.scrapers.browser.agent.tools import AgentDecision, get_decision_schema
from jet.scrapers.browser.agent.utils import get_step_dir, save_json, save_text
from rich.console import Console

console = Console()
SYSTEM_PROMPT_TEMPLATE = """
You are an autonomous browser agent powered by Playwright. Your goal is to: {goal}
Available Tools:
{tools_json}
CRITICAL RULES FOR INTERACTION:
1. USE REFS DIRECTLY: When the snapshot shows an element like `- textbox "Job title" [ref=e39]`, use "e39" as the target.
2. IMMEDIATE FINAL ANSWER: As soon as you have the information required by the goal, call `final_answer` immediately.
3. NO REDUNDANT ACTIONS: If a form is already filled or a button has been clicked, do not repeat the action. Check the snapshot state carefully.
4. EFFICIENT FORM FILLING: Use `browser_fill_form` to fill multiple fields at once.
5. STRUCTURED OUTPUT: If the goal asks for structured data (like JSON), pass the raw JSON object/array directly to the `result` argument of `final_answer`.
6. EFFICIENCY: Choose the most direct path to the goal. Avoid unnecessary navigation or waiting.
RESPONSE FORMAT:
You must respond using the provided JSON schema. Ensure all required arguments for the chosen tool are present.
"""
USER_PROMPT_TEMPLATE = """
Current URL: {url}
Previous URL: {prev_url}
Current Page Snapshot:
{snapshot}
Previous Action Result:
{last_result}
"""


class LocalAgent:
    def __init__(
        self,
        api_base: str,
        model: str,
        registry: PlaywrightToolRegistry,
        output_dir: Path,
        max_snapshot_chars: int = 10000,
    ):
        self.client = openai.AsyncOpenAI(base_url=api_base, api_key="local")
        self.model = model
        self.registry = registry
        self.output_dir = output_dir
        self.max_snapshot_chars = max_snapshot_chars
        self.prev_url = "Unknown"  # Track previous URL across steps

    async def think_and_act(
        self, goal: str, initial_snapshot: str, max_steps: int = 15, **kwargs
    ) -> str:
        tools_json = json.dumps(self.registry.get_tool_schema(), indent=2)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(goal=goal, tools_json=tools_json)
        last_result = "None"
        current_state = initial_snapshot
        # Initialize prev_url from toolbox
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
            save_text(step_dir / "input_snapshot.md", current_state)
            try:
                await self.registry.toolbox.browser_take_screenshot(
                    filename=f"step_{step_index}_start.png", step_dir=step_dir
                )
            except Exception as e:
                console.print(
                    f"[yellow]Failed to capture start screenshot: {e}[/yellow]"
                )
            user_prompt_content = USER_PROMPT_TEMPLATE.format(
                url=current_url,
                prev_url=self.prev_url,
                snapshot=current_state[: self.max_snapshot_chars],
                last_result=last_result,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_content},
            ]
            self._save_step_input(
                step_dir, system_prompt, user_prompt_content, messages, current_state
            )
            try:
                decision = await self._call_llm(
                    messages, response_format, step_dir, **kwargs
                )
                if decision.tool == "final_answer":
                    decision.finished = True
                if decision.finished:
                    console.print("[bold green]Goal achieved.[/bold green]")
                    return "Goal achieved."
                console.print(f"\n[dim]Thought: {decision.thought}[/dim]")
                console.print(f"[bold yellow]Executing:[/bold yellow] {decision.tool}")
                last_result = await self._execute_tool(decision, step_dir)
                # Update prev_url after executing tool
                self.prev_url = current_url
                if not self.registry.toolbox.is_closed:
                    try:
                        output_state = await self.registry.toolbox.browser_snapshot(
                            step_dir=step_dir
                        )
                        save_text(step_dir / "output_snapshot.md", output_state)
                        await self.registry.toolbox.browser_take_screenshot(
                            filename=f"step_{step_index}_end.png", step_dir=step_dir
                        )
                    except Exception as e:
                        console.print(
                            f"[yellow]Failed to capture end state: {e}[/yellow]"
                        )
                if self.registry.toolbox.is_closed:
                    return "Browser closed."
            except Exception as e:
                console.print(f"[bold red]Error in step {step_index}:[/bold red] {e}")
                last_result = f"Error: {str(e)}"
                save_text(step_dir / "error.log", str(e))
        console.print("[bold yellow]Max steps reached.[/bold yellow]")
        return "Max steps reached."

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
