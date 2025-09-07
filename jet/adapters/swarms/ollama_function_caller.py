import json
import re

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Type

from pydantic import BaseModel
import ollama


SUPPORTED_OLLAMA_MODELS = [
    "llama3.2",
    "llama3.1",
    "mistral",
    "codellama",
]


def check_ollama_host():
    return os.getenv("OLLAMA_HOST", "http://localhost:11434")


class OllamaFunctionCaller:
    """
    A class to interact with the Ollama API for generating text based on a system prompt and a task.
    """

    def __init__(
        self,
        system_prompt: str,
        base_model: Optional[Type[BaseModel]] = None,
        temperature: float = 0.1,
        max_tokens: int = 5000,
        model_name: str = "llama3.2",
    ):
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.base_model = base_model
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.host = check_ollama_host()

    def run(self, task: str):
        try:
            # merge schema hint into single system message if schema is provided
            system_message = self.system_prompt
            if self.base_model:
                system_message += (
                    f"\nReturn the response as a JSON object containing only the data fields defined in the following schema, "
                    f"without including the schema itself or any additional metadata:\n"
                    f"{self.base_model.model_json_schema()}\n"
                    f"For example, if the schema defines fields 'name' and 'age', return only {{'name': 'value', 'age': number}}."
                )

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": task},
            ]

            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": self.temperature,
                         "num_predict": self.max_tokens},
            )

            response_text = response["message"]["content"]

            if self.base_model:
                # Clean response to extract valid JSON
                cleaned_response = re.sub(r'^```json\n|```$', '', response_text.strip())
                try:
                    json_data = json.loads(cleaned_response)
                    # Extract nested 'properties' if present (for backward compatibility)
                    if isinstance(json_data, dict) and 'properties' in json_data:
                        json_data = json_data['properties']
                    return self.base_model.model_validate(json_data)
                except json.JSONDecodeError as e:
                    print(f"Invalid JSON response: {e}")
                    return None

            return response_text

        except Exception as e:
            print(f"There was an error: {e}")

    def check_model_support(self):
        for model in SUPPORTED_OLLAMA_MODELS:
            print(model)
        return SUPPORTED_OLLAMA_MODELS

    def batch_run(self, tasks: List[str]) -> List:
        return [self.run(task) for task in tasks]

    def concurrent_run(self, tasks: List[str]) -> List:
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            return list(executor.map(self.run, tasks))
