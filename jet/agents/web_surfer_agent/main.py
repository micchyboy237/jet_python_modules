"""
Web Surfer Agent

Usage Examples:

1. Find the top story on Hacker News:
   python -m jet.agents.web_surfer_agent.main "Find the top story on Hacker News and tell me its title."

2. Search for a specific topic on Google:
   python -m jet.agents.web_surfer_agent.main "Search for 'OpenTelemetry best practices' on Google and summarize the first result."

3. Fill out a form (hypothetical):
   python -m jet.agents.web_surfer_agent.main "Go to example.com/form, fill 'Name' with 'John', 'Email' with 'john@example.com', and submit."
"""

import argparse
import asyncio
import json
from typing import Dict, List

from jet.adapters.llama_cpp.config import LLM_MODEL, PHOENIX_BASE_URL
from jet.adapters.llama_cpp.llm_utils_observed import achat
from jet.libs.llama_cpp.usage.chat_stream_types import StreamCompletionResult
from jet_telemetry import agent, chain, initialize_telemetry
from playwright.async_api import async_playwright

from . import tools


class LocalWebSurfer:
    def __init__(self, model_name: str = LLM_MODEL, max_steps: int = 15):
        self.model_name = model_name
        self.max_steps = max_steps
        self.history: List[Dict] = []
        self.page = None

        # Map tool names to their schema definitions for the LLM
        self.tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "browser_navigate",
                    "description": "Navigate the browser to a specified URL.",
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
                    "name": "browser_navigate_back",
                    "description": "Go back in browser history.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_snapshot",
                    "description": "Capture accessibility tree of current page as structured YAML.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_find",
                    "description": "Search the snapshot for text or regex patterns.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "Text or regex pattern to find",
                            }
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_click",
                    "description": "Click an element by its accessible role and name.",
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
                    "name": "browser_hover",
                    "description": "Hover over an element by its accessible role and name.",
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
                    "description": "Type text into an input field by its accessible label.",
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
                    "name": "browser_fill_form",
                    "description": "Fill multiple form fields at once.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fields": {
                                "type": "object",
                                "description": "Dictionary of label->value pairs",
                            }
                        },
                        "required": ["fields"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_select_option",
                    "description": "Select dropdown option.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": "Label of the select element",
                            },
                            "option": {
                                "type": "string",
                                "description": "Option value or text to select",
                            },
                        },
                        "required": ["label", "option"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_press_key",
                    "description": "Press a key (Enter, Tab, etc.).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "description": "Key to press (e.g., 'Enter', 'Tab')",
                            }
                        },
                        "required": ["key"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_take_screenshot",
                    "description": "Take PNG screenshot of current page.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Optional path to save screenshot",
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_tabs",
                    "description": "List, create, close, or select tabs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["list", "new", "close", "select"],
                                "description": "Action to perform",
                            },
                            "index": {
                                "type": "integer",
                                "description": "Tab index (for close/select)",
                            },
                        },
                        "required": ["action"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_handle_dialog",
                    "description": "Accept or dismiss JavaScript dialogs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "accept": {
                                "type": "boolean",
                                "description": "Whether to accept the dialog",
                            },
                            "prompt_text": {
                                "type": "string",
                                "description": "Text to enter if prompted",
                            },
                        },
                        "required": ["accept"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_evaluate",
                    "description": "Evaluate JavaScript on page.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "script": {
                                "type": "string",
                                "description": "JavaScript code to evaluate",
                            }
                        },
                        "required": ["script"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_wait_for",
                    "description": "Wait for text or time.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to wait for",
                            },
                            "timeout_ms": {
                                "type": "integer",
                                "description": "Timeout in milliseconds",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_close",
                    "description": "Close the page.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

    async def initialize(self):
        """Start the Playwright browser."""
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=False)
        self.page = await browser.new_page()
        tools.set_page(self.page)
        print("[INFO] Browser initialized.")

    @chain(name="plan_and_act_step")
    async def run_step(self, messages: list, step_num: int) -> tuple[list, bool]:
        """
        Executes a single step of the agent loop: Observe -> Plan -> Act.
        Returns updated messages and a boolean indicating if the task is done.
        """
        print(f"\n--- Step {step_num} ---")

        # 1. Observe
        observation = await tools.browser_snapshot()
        messages.append(
            {"role": "user", "content": f"Current Page State:\n{observation}"}
        )

        print(f"[THINKING] Asking {self.model_name} for next move...")

        # 2. Plan (LLM Call)
        response: StreamCompletionResult = await achat(
            prompt_or_messages=messages,
            model=self.model_name,
            tools=self.tools_schema,
            tool_registry=None,  # We handle execution manually to access self.page context via tools module
            project_name="web-surfer-agent",
        )

        messages.append({"role": "assistant", "content": response.content})

        # 3. Act
        if response.has_tool_calls:
            for tc in response.tool_calls:
                print(f"[ACTION] Calling {tc.name} with {tc.arguments}")

                # Resolve function from tools module
                func = getattr(tools, tc.name, None)
                if not func:
                    result = f"[ERROR] Unknown tool: {tc.name}"
                else:
                    try:
                        # Ensure args are dict
                        args = tc.arguments
                        if isinstance(args, str):
                            args = json.loads(args)

                        # Call the async tool function
                        result = await func(**args)
                    except Exception as e:
                        result = f"[ERROR] Execution of {tc.name} failed: {str(e)}"

                print(f"[RESULT] {result}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": result,
                    }
                )
            return messages, False  # Continue loop
        else:
            print(f"[FINAL ANSWER]: {response.content}")
            return messages, True  # Task done

    @agent(name="web-surfer-agent")
    async def run(self, goal: str):
        """Main agentic loop: Observe -> Decide (LLM) -> Act."""
        await self.initialize()

        messages = [
            {
                "role": "user",
                "content": f"Your goal is to: {goal}. Use the provided tools to achieve it. Do not stop until the goal is met or you determine it cannot be met.",
            }
        ]

        for step in range(self.max_steps):
            messages, is_done = await self.run_step(messages, step + 1)
            if is_done:
                break

        print("\n[SUCCESS] Agent finished.")


def get_args():
    parser = argparse.ArgumentParser(
        description="Web Surfer Agent: Autonomous browser agent with LLM planning."
    )
    parser.add_argument(
        "goal",
        type=str,
        help="The goal for the agent to achieve (e.g., 'Find the top story on Hacker News').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=LLM_MODEL,
        help=f"LLM model to use (default: {LLM_MODEL}).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Maximum number of steps/iterations for the agent.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (default: visible).",
    )
    return parser.parse_args()


async def main():
    args = get_args()

    # Initialize telemetry
    initialize_telemetry(service_name="web-surfer-agent", endpoint=PHOENIX_BASE_URL)

    agent = LocalWebSurfer(model_name=args.model, max_steps=args.max_steps)

    # Override headless if needed (requires passing to initialize, simplified here for demo)
    # In a real refactor, initialize would accept headless param

    await agent.run(args.goal)


if __name__ == "__main__":
    asyncio.run(main())
