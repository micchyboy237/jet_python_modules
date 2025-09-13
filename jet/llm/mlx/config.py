import os
from jet.models.model_types import LLMModelType
from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name


DEFAULT_MODEL: LLMModelType = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_LOG_DIR = os.path.expanduser(
    f"{get_entry_file_dir()}/generated/{os.path.splitext(get_entry_file_name())[0]}/llm"
)
DEFAULT_OLLAMA_LOG_DIR = os.path.expanduser(
    f"{get_entry_file_dir()}/generated/{os.path.splitext(get_entry_file_name())[0]}/llm"
)
