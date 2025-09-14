import asyncio
import os
import shutil
import yaml
import warnings
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ModelFamily
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core.models import ChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.messages import TextMessage

from jet.logger import logger

CWD = os.path.dirname(__file__)
os.chdir(CWD)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated")
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file =  f"{OUTPUT_DIR}/main.log"
logger.basicConfig(filename=log_file)
logger.orange(f"Logs: {log_file}")
logger.info(f"Current Working Dir: {CWD}")

WORK_DIR = f"{OUTPUT_DIR}/coding"

# Suppress warnings about the requests.Session() not being closed
warnings.filterwarnings(
    action="ignore", message="unclosed", category=ResourceWarning)


async def main() -> None:

    # Load model configuration and create the model client.
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    orchestrator_client = ChatCompletionClient.load_component(
        config["orchestrator_client"])
    coder_client = ChatCompletionClient.load_component(config["coder_client"])
    web_surfer_client = ChatCompletionClient.load_component(
        config["web_surfer_client"])
    file_surfer_client = ChatCompletionClient.load_component(
        config["file_surfer_client"])

    # Read the prompt
    prompt = ""
    with open("prompt.txt", "rt") as fh:
        prompt = fh.read().strip()
    filename = "__FILE_NAME__".strip()

    # Set up the team
    coder = MagenticOneCoderAgent(
        "Assistant",
        model_client=coder_client,
        model_client_stream=True,
    )

    executor = CodeExecutorAgent(
        "ComputerTerminal",
        code_executor=LocalCommandLineCodeExecutor(
            work_dir=WORK_DIR,
            cleanup_temp_files=False,
        )
    )

    file_surfer = FileSurfer(
        name="FileSurfer",
        model_client=file_surfer_client,
        base_path=CWD,
    )

    web_surfer = MultimodalWebSurfer(
        name="WebSurfer",
        model_client=web_surfer_client,
        downloads_folder=os.path.join(OUTPUT_DIR, "downloads"),
        debug_dir=os.path.join(OUTPUT_DIR, "logs"),
        browser_data_dir=os.path.join(OUTPUT_DIR, "browser_data"),
        to_save_screenshots=True,
    )

    team = MagenticOneGroupChat(
        [coder, executor, file_surfer, web_surfer],
        model_client=orchestrator_client,
        max_turns=20,
        final_answer_prompt=f""",
We have completed the following task:

{prompt}

The above messages contain the conversation that took place to complete the task.
Read the above conversation and output a FINAL ANSWER to the question.
To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and don't include units such as $ or percent signs unless specified otherwise.
If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
""".strip()
    )

    # Prepare the prompt
    filename_prompt = ""
    if len(filename) > 0:
        filename_prompt = f"The question is about a file, document or image, which can be accessed by the filename '{filename}' in the current working directory."
    task = f"{prompt}\n\n{filename_prompt}"

    # Run the task
    stream = team.run_stream(task=task.strip())
    await Console(stream)

if __name__ == "__main__":
    asyncio.run(main())
