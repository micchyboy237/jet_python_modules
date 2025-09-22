import os
import shutil
from typing import List, Dict, Any
from pathlib import Path
import json

from ollama import Client, Message

from jet.file.utils import save_file
from jet.transformers.formatters import format_json
from jet.logger import logger

# Constants
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
SYSTEM_MESSAGE = """
You are an expert Python developer tasked with updating code to replace OpenAI LLM calls with Ollama's LLaMA 3.2 model. Ensure the modified code:
- Uses the Ollama Python client (already imported).
- Maintains the original functionality and structure.
- Follows industry best practices (DRY, modular, testable).
- Includes type hints where appropriate.
- Preserves existing imports unless explicitly replacing OpenAI.
- Provides clear comments explaining changes.
When modifying code, return only the updated code content without additional explanations unless requested.
"""

class CodeTransformer:
    """Handles code transformation using Ollama LLM."""

    def __init__(self, model: str, host: str = 'http://localhost:11435'):
        self.client = Client(host=host)
        self.model = model

    def create_prompt(self, file_path: str, user_instruction: str) -> List[Message]:
        """
        Creates a structured prompt for code transformation.
        
        Args:
            file_path: Path to the source code file
            user_instruction: User-provided transformation instruction
            
        Returns:
            List of message dictionaries for Ollama
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                code_content = file.read()
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise

        return [
            {'role': 'system', 'content': SYSTEM_MESSAGE},
            {
                'role': 'user',
                'content': f"""
# Original Code
```python
{code_content}
```

# Instruction
{user_instruction}
"""
            }
        ]

    def transform_code(self, file_path: str, instruction: str) -> str:
        """
        Transforms code using Ollama LLM based on the provided instruction.
        
        Args:
            file_path: Path to the source code file
            instruction: Transformation instruction
            
        Returns:
            Transformed code as a string
        """
        messages = self.create_prompt(file_path, instruction)
        response_text = ""
        try:
            for part in self.client.chat(model=self.model, messages=messages, stream=True):
                content = part['message']['content']
                logger.teal(content, flush=True)
                response_text += content
        except Exception as e:
            logger.error(f"Error during LLM transformation: {str(e)}")
            raise
        return response_text

def setup_output_directory() -> None:
    """Sets up the output directory and logging configuration."""
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger.basicConfig(filename=log_file)
    logger.info(f"Logs: {log_file}")

def main():
    setup_output_directory()

    transformer = CodeTransformer(model='qwen2.5-coder:7b-instruct-q4_K_M')
    file_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/llm/evaluators/rag_evaluator.py"
    instruction = "Update this code to replace openai LLM with ollama model llama3.2"

    try:
        response_text = transformer.transform_code(file_path, instruction)

        save_file(instruction, f"{OUTPUT_DIR}/query.md")
        save_file(response_text, f"{OUTPUT_DIR}/response.md")
        save_file(format_json({'response': response_text}), f"{OUTPUT_DIR}/output.json")
    except Exception as e:
        logger.error(f"Failed to process transformation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
