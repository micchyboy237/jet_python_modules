from typing import Literal, Union, Dict, List, Optional, TypedDict, Any, get_args
from enum import Enum
from jet.llm.mlx.helpers.detect_repetition import NgramRepeat
from transformers import PreTrainedTokenizer
from mlx_lm.tokenizer_utils import TokenizerWrapper

# Type definitions

MLXTokenizer = Union[TokenizerWrapper, PreTrainedTokenizer]


class Message(TypedDict):
    role: str
    content: str


class Delta(TypedDict):
    role: Optional[str]
    content: Optional[str]


class Tool(TypedDict):
    type: str
    function: Dict[str, Any]


class RoleMapping(TypedDict, total=False):
    system_prompt: str
    system: str
    user: str
    assistant: str
    stop: str


class Logprobs(TypedDict):
    token_logprobs: List[float]
    top_logprobs: List[Dict[int, float]]
    tokens: List[int]


class Choice(TypedDict):
    index: int
    logprobs: Logprobs
    finish_reason: Optional[Literal["length", "stop"]]
    message: Optional[Message]
    delta: Optional[Delta]
    text: Optional[str]


class Usage(TypedDict):
    prompt_tokens: int
    prompt_tps: float
    completion_tokens: int
    completion_tps: float
    total_tokens: int
    peak_memory: float


class ModelInfo(TypedDict):
    id: str
    object: str
    created: int


class ModelsResponse(TypedDict):
    object: str
    data: List[ModelInfo]


class ModelTypeEnum(Enum):
    DOLPHIN3_LLAMA3_8B_4BIT = "dolphin3.0-llama3.1-8b-4bit"
    LLAMA_3_1_8B_INSTRUCT_4BIT = "llama-3.1-8b-instruct-4bit"
    LLAMA_3_2_1B_INSTRUCT_4BIT = "llama-3.2-1b-instruct-4bit"
    LLAMA_3_2_3B_INSTRUCT_4BIT = "llama-3.2-3b-instruct-4bit"
    MISTRAL_NEMO_INSTRUCT_2407_4BIT = "mistral-nemo-instruct-2407-4bit"
    QWEN2_5_7B_INSTRUCT_4BIT = "qwen2.5-7b-instruct-4bit"
    QWEN2_5_14B_INSTRUCT_4BIT = "qwen2.5-14b-instruct-4bit"
    QWEN2_5_CODER_14B_INSTRUCT_4BIT = "qwen2.5-coder-14b-instruct-4bit"
    QWEN3_0_6B_4BIT = "qwen3-0.6b-4bit"
    QWEN3_1_7B_4BIT = "qwen3-1.7b-4bit"
    QWEN3_4B_4BIT = "qwen3-4b-4bit"
    QWEN3_8B_4BIT = "qwen3-8b-4bit"

    @property
    def key(self) -> str:
        return self.value

    @property
    def model_path(self) -> str:
        return f"mlx-community/{self.value.replace('.', '.').replace('_', '-')}"


# Model key types
LLMModelKey = Literal[
    "dolphin3.0-llama3.1-8b-4bit",
    "llama-3.1-8b-instruct-4bit",
    "llama-3.2-1b-instruct-4bit",
    "llama-3.2-3b-instruct-4bit",
    "mistral-nemo-instruct-2407-4bit",
    "qwen2.5-7b-instruct-4bit",
    "qwen2.5-14b-instruct-4bit",
    "qwen2.5-coder-14b-instruct-4bit",
    "qwen3-0.6b-4bit",
    "qwen3-embedding-0.6b-4bit",
    "qwen3-1.7b-4bit",
    "qwen3-4b-4bit",
    "qwen3-8b-4bit"
]

# Model value types
LLMModelValue = Literal[
    "mlx-community/Dolphin3.0-Llama3.1-8B-4bit",
    "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Qwen2.5-14B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
    "mlx-community/Qwen3-0.6B-4bit-DWQ-053125",
    "mlx-community/Qwen3-1.7B-4bit-DWQ-053125",
    "mlx-community/Qwen3-4B-4bit-DWQ-053125",
    "mlx-community/Qwen3-8B-4bit-DWQ-053125"
]


class EmbedModelTypeEnum(Enum):
    EMBED_ALL_MINILM_L6_V2_BF16 = "all-minilm-l6-v2-bf16"
    EMBED_ALL_MINILM_L6_V2_8BIT = "all-minilm-l6-v2-8bit"
    EMBED_ALL_MINILM_L6_V2_6BIT = "all-minilm-l6-v2-6bit"
    EMBED_ALL_MINILM_L6_V2_4BIT = "all-minilm-l6-v2-4bit"

    @property
    def key(self) -> str:
        return self.value

    @property
    def model_path(self) -> str:
        return f"mlx-community/{self.value.replace('.', '.').replace('_', '-')}"


# Embed model key types
EmbedModelKey = Literal[
    # HF
    "all-mpnet-base-v2",
    "all-MiniLM-L12-v2",
    "all-MiniLM-L6-v2",
    "paraphrase-MiniLM-L12-v2",
    # Ollama
    "nomic-embed-text",
    "mxbai-embed-large",
    "granite-embedding",
    "granite-embedding:278m",
    "all-minilm:22m",
    "all-minilm:33m",
    "snowflake-arctic-embed:33m",
    "snowflake-arctic-embed-m",
    "snowflake-arctic-embed:137m",
    "snowflake-arctic-embed",
    "paraphrase-multilingual",
    "bge-large",
    # MLX
    "all-minilm-l6-v2-bf16",
    "all-minilm-l6-v2-8bit",
    "all-minilm-l6-v2-6bit",
    "all-minilm-l6-v2-4bit"
]

# Embed model value types
EmbedModelValue = Literal[
    # HF
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-MiniLM-L12-v2",
    # OLLAMA
    "nomic-ai/nomic-embed-text-v1.5",
    "mixedbread-ai/mxbai-embed-large-v1",
    "ibm-granite/granite-embedding-30m-english",
    "ibm-granite/granite-embedding-278m-multilingual",
    "Snowflake/snowflake-arctic-embed-s",
    "Snowflake/snowflake-arctic-embed-m",
    "Snowflake/snowflake-arctic-embed-m-long",
    "Snowflake/snowflake-arctic-embed-l",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "BAAI/bge-large-en-v1.5",
    # MLX
    "mlx-community/all-MiniLM-L6-v2-bf16",
    "mlx-community/all-MiniLM-L6-v2-8bit",
    "mlx-community/all-MiniLM-L6-v2-6bit",
    "mlx-community/all-MiniLM-L6-v2-4bit"
]

# Combined llm model type
# LLMModelType = Union[LLMModelKey, LLMModelValue, ModelTypeEnum]
LLMModelType = Union[LLMModelKey, LLMModelValue, ModelTypeEnum]

# Combined embed model type
EmbedModelType = Union[EmbedModelKey, EmbedModelValue, EmbedModelTypeEnum]

ModelKey = Union[LLMModelKey, EmbedModelKey]
ModelValue = Union[LLMModelValue, EmbedModelValue]
ModelType = Union[LLMModelType, EmbedModelType]


# Extract Enum classes from the union
enum_types_in_model_type = [t for t in get_args(
    ModelType) if isinstance(t, type) and issubclass(t, Enum)]

# All model keys from LLMModelKey + EmbedModelKey inside ModelKey union
model_keys_list: List[str] = [
    arg for arg in get_args(ModelKey) if isinstance(arg, str)
]


# All model values from LLMModelValue + EmbedModelValue inside ModelValue union
model_values_list: List[str] = [
    arg for arg in get_args(ModelValue) if isinstance(arg, str)
]


# All model types: keys + values + Enum values from ModelTypeEnum and EmbedModelTypeEnum
model_types_list: List[str] = (
    [arg for arg in get_args(ModelType) if isinstance(arg, str)] +
    [e.value for e in ModelTypeEnum] +
    [e.value for e in EmbedModelTypeEnum]
)


class CompletionResponse(TypedDict):
    id: str
    system_fingerprint: str
    object: str
    model: LLMModelType
    created: int
    usage: Optional[Usage]
    content: str
    repetitions: Optional[List[NgramRepeat]]
    choices: List[Choice]
