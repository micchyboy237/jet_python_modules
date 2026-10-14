import asyncio
import json
from typing import Dict, List

from jet.adapters.llama_cpp.config import LLM_MODEL, PHOENIX_BASE_URL
from jet.adapters.llama_cpp.llm_utils_observed import achat
from jet.libs.llama_cpp.usage.chat_stream_types import StreamCompletionResult
from jet_telemetry import agent, initialize_telemetry, performance_monitor, tool
from playwright.async_api import Page, async_playwright

# Initialize telemetry for this agent
initialize_telemetry(service_name="web-surfer-agent", endpoint=PHOENIX_BASE_URL)


def browser_navigate(url: str) -> str:
    """Navigate the browser to a specific URL."""
    return f"Navigated to {url}"


def browser_click(role: str, name: str) -> str:
    """Click an element identified by its ARIA role and accessible name."""
    return f"Clicked element with role '{role}' and name '{name}'"


def browser_type(label: str, text: str) -> str:
    """Type text into a form field identified by its label."""
    return f"Typed '{text}' into field labeled '{label}'"


def browser_extract(selector: str) -> str:
    """Extract text content from elements matching a CSS selector."""
    return f"Extracted content using selector '{selector}'"


class LocalWebSurfer:
    def __init__(self, model_name: str = LLM_MODEL, max_steps: int = 15):
        self.model_name = model_name
        self.max_steps = max_steps
        self.history: List[Dict] = []
        self.page: Page = None
        self.tools_map = {
            "browser_navigate": browser_navigate,
            "browser_click": browser_click,
            "browser_type": browser_type,
            "browser_extract": browser_extract,
        }

    async def initialize(self):
        """Start the Playwright browser."""
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=False)
        self.page = await browser.new_page()
        print("[INFO] Browser initialized.")

    @tool(
        name="observe_page",
        description="Captures the current page state including URL, title, and ARIA snapshot.",
    )
    async def observe(self) -> str:
        """Capture the current page state using the accessibility tree."""
        try:
            snapshot = await self.page.locator("body").aria_snapshot()
            return f"URL: {self.page.url}\nTitle: {await self.page.title()}\n\nAccessibility Tree:\n{snapshot}"
        except Exception as e:
            return f"[ERROR] Observation failed: {str(e)}"

    @tool(description="Executes a browser action based on LLM tool call.")
    @performance_monitor(threshold_ms=2000)
    async def execute_action(self, tool_call) -> str:
        """Translate an LLM tool call into a Playwright action."""
        func_name = tool_call.name
        args = tool_call.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return f"[ERROR] Failed to parse arguments for {func_name}"

        func = self.tools_map.get(func_name)
        if not func:
            return f"[ERROR] Unknown tool: {func_name}"

        # Set specific tool name attribute for better visibility in traces
        # Note: The @tool decorator sets the generic name, but we can't easily
        # change it dynamically per call without complex wrappers.
        # However, the parameters will show which action was taken.

        try:
            if func_name == "browser_navigate":
                await self.page.goto(args["url"], wait_until="domcontentloaded")
                return func(**args)
            elif func_name == "browser_click":
                locator = self.page.get_by_role(args["role"], name=args["name"])
                await locator.click(timeout=5000)
                await self.page.wait_for_load_state("domcontentloaded")
                return func(**args)
            elif func_name == "browser_type":
                locator = self.page.get_by_label(args["label"])
                await locator.fill(args["text"])
                return func(**args)
            elif func_name == "browser_extract":
                elements = await self.page.query_selector_all(args["selector"])
                texts = [await el.inner_text() for el in elements[:5]]
                return json.dumps(texts)
        except Exception as e:
            return f"[ERROR] Execution of {func_name} failed: {str(e)}"

        return "[ERROR] Execution failed."

    @agent(name="web-surfer-agent")
    async def run(self, goal: str):
        """Main agentic loop: Observe -> Decide (LLM) -> Act."""
        await self.initialize()

        # Define tools in OpenAI format for llm_utils_observed
        tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "browser_navigate",
                    "description": "Navigate the browser to a specific URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL to navigate to",
                            }
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_click",
                    "description": "Click an element identified by its ARIA role and accessible name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "type": "string",
                                "description": "ARIA role of the element",
                            },
                            "name": {
                                "type": "string",
                                "description": "Accessible name of the element",
                            },
                        },
                        "required": ["role", "name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_type",
                    "description": "Type text into a form field identified by its label.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": "Label of the form field",
                            },
                            "text": {"type": "string", "description": "Text to type"},
                        },
                        "required": ["label", "text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_extract",
                    "description": "Extract text content from elements matching a CSS selector.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {
                                "type": "string",
                                "description": "CSS selector",
                            }
                        },
                        "required": ["selector"],
                    },
                },
            },
        ]

        messages = [
            {
                "role": "user",
                "content": f"Your goal is to: {goal}. Use the provided tools to achieve it. Do not stop until the goal is met or you determine it cannot be met.",
            }
        ]

        for step in range(self.max_steps):
            print(f"\n--- Step {step + 1} ---")

            # 1. Observe
            observation = await self.observe()
            messages.append(
                {"role": "user", "content": f"Current Page State:\n{observation}"}
            )

            print(f"[THINKING] Asking {self.model_name} for next move...")

            # 2. Plan (LLM Call)
            # achat creates its own LLM span internally via llm_utils_observed
            response: StreamCompletionResult = await achat(
                prompt_or_messages=messages,
                model=self.model_name,
                tools=tools_schema,
                tool_registry=None,  # We handle execution manually to access self.page
                project_name="web-surfer-agent",
            )

            messages.append({"role": "assistant", "content": response.content})

            # 3. Act
            if response.has_tool_calls:
                for tc in response.tool_calls:
                    print(f"[ACTION] Calling {tc.name} with {tc.arguments}")
                    result = await self.execute_action(tc)
                    print(f"[RESULT] {result}")

                    # Append tool result to messages for next turn
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": result,
                        }
                    )
            else:
                print(f"[FINAL ANSWER]: {response.content}")
                break

        print("\n[SUCCESS] Agent finished.")


async def main():
    agent = LocalWebSurfer(model_name=LLM_MODEL)
    await agent.run("Find the top story on Hacker News and tell me its title.")


if __name__ == "__main__":
    asyncio.run(main())
