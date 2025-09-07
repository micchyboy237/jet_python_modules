import json
import re

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Type

from pydantic import BaseModel
# from mlx_lm import load, generate
# from mlx_lm.sample_utils import make_sampler
from jet.llm.mlx.remote import generation as gen
from jet.llm.mlx.remote.types import Message


SUPPORTED_MLX_MODELS = [
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
]


class MLXFunctionCaller:
    """
    A class to interact with MLX local models for generating text based on a system prompt and a task.
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        base_model: Optional[Type[BaseModel]] = None,
        temperature: float = 0.1,
        max_tokens: int = 5000,
        model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        tools: Optional[List[Callable]] = None
    ):
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.base_model = base_model
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.tools = tools
        # self.model, self.tokenizer = load(self.model_name)

    def run(self, task: str):
        try:
            system_message = self.system_prompt or ""
            if self.base_model:
                system_message += (
                    f"\nReturn the response as a JSON object containing only the data fields defined in the following schema, "
                    f"without including the schema itself or any additional metadata:\n"
                    f"{self.base_model.model_json_schema()}\n"
                    f"For example, if the schema defines fields 'name' and 'age', return only {{\"name\": \"value\", \"age\": number}}."
                )
            messages: List[Message] = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": task})
            chunks = gen.stream_chat(
                messages=messages,
                model=self.model_name,  # Explicitly pass model_name
                temperature=self.temperature,
                tools=self.tools,
                verbose=True,
                max_tokens=self.max_tokens,
            )
            response_text = ""
            for chunk in chunks:
                response_text += chunk["choices"][0]["message"]["content"]
            if self.base_model:
                cleaned_response = re.sub(r'^```json\n|```$', '', response_text, flags=re.MULTILINE)
                try:
                    json_data = json.loads(cleaned_response)
                    if isinstance(json_data, dict) and 'properties' in json_data:
                        json_data = json_data['properties']
                    return self.base_model.model_validate(json_data)
                except json.JSONDecodeError as e:
                    print(f"Invalid JSON response: {e}")
                    return None
            return response_text
        except Exception as e:
            print(f"There was an error: {e}")
            return None
    def check_model_support(self):
        for model in SUPPORTED_MLX_MODELS:
            print(model)
        return SUPPORTED_MLX_MODELS

    def batch_run(self, tasks: List[str]) -> List:
        return [self.run(task) for task in tasks]

    def concurrent_run(self, tasks: List[str]) -> List:
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            return list(executor.map(self.run, tasks))
