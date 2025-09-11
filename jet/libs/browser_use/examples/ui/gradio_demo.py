# pyright: reportMissingImports=false
from typing import Optional
from jet.adapters.browser_use.ollama.chat import ChatOllama
from browser_use import Agent, BrowserProfile
from rich.text import Text
from rich.panel import Panel
from rich.console import Console
import gradio as gr  # type: ignore
from dotenv import load_dotenv
import asyncio
import os
import sys
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


load_dotenv()

# Third-party imports

# Local module imports


@dataclass
class ActionResult:
    is_done: bool
    extracted_content: str | None
    error: str | None
    include_in_memory: bool


@dataclass
class AgentHistoryList:
    all_results: list[ActionResult]
    all_model_outputs: list[dict]


def parse_agent_history(history_str: str) -> None:
    console = Console()

    # Split the content into sections based on ActionResult entries
    sections = history_str.split('ActionResult(')

    for i, section in enumerate(sections[1:], 1):  # Skip first empty section
        # Extract relevant information
        content = ''
        if 'extracted_content=' in section:
            content = section.split('extracted_content=')[
                1].split(',')[0].strip("'")

        if content:
            header = Text(f'Step {i}', style='bold blue')
            panel = Panel(content, title=header, border_style='blue')
            console.print(panel)
            console.print()

    return None


async def run_browser_task(
        task: str,
        model: str = 'llama3.2',
        headless: bool = True,
        api_key: Optional[str] = None,
) -> str:
    # if not api_key.strip():
    #     return 'Please provide an API key'

    # os.environ['OPENAI_API_KEY'] = api_key

    browser_profile = BrowserProfile(headless=headless)
    agent = Agent(
        task=task,
        llm=ChatOllama(model=model),
        browser_profile=browser_profile,
    )
    result = await agent.run()
    #  TODO: The result could be parsed better
    return str(result)


def create_ui():
    with gr.Blocks(title='Browser Use GUI') as interface:
        gr.Markdown('# Browser Use Task Automation')

        with gr.Row():
            with gr.Column():
                api_key = gr.Textbox(label='OpenAI API Key',
                                     placeholder='sk-...', type='password')
                task = gr.Textbox(
                    label='Task Description',
                    placeholder='E.g., Find flights from New York to London for next week',
                    lines=3,
                )
                model = gr.Dropdown(
                    choices=[
                        'nomic-embed-text',
                        'mxbai-embed-large',
                        'all-minilm:22m',
                        'all-minilm:33m',
                        'paraphrase-multilingual',
                        'bge-large',
                        'granite-embedding',
                        'granite-embedding:278m',
                    ],
                    label='Model',
                    value='nomic-embed-text'
                )
                headless = gr.Checkbox(label='Run Headless', value=True)
                submit_btn = gr.Button('Run Task')

            with gr.Column():
                output = gr.Textbox(
                    label='Output', lines=10, interactive=False)

        submit_btn.click(
            fn=lambda *args: asyncio.run(run_browser_task(*args)),
            inputs=[task, model, headless, api_key],
            outputs=output,
        )

    return interface


if __name__ == '__main__':
    demo = create_ui()
    demo.launch()
