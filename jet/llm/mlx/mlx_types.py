from typing import Literal, Union
from enum import Enum
from typing import Dict, List, Optional, Union, Literal, TypedDict, Any, Iterator

# Type definitions


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


class CompletionResponse(TypedDict):
    id: str
    system_fingerprint: str
    object: str
    model: str
    created: int
    choices: List[Choice]
    usage: Optional[Usage]


class ModelInfo(TypedDict):
    id: str
    object: str
    created: int


class ModelsResponse(TypedDict):
    object: str
    data: List[ModelInfo]


class ModelTypeEnum(Enum):
    DOLPHIN3_LLAMA3_8B_4BIT = "dolphin3.0-llama3.1-8b-4bit"
    GEMMA_3_1B_IT_QAT_4BIT = "gemma-3-1b-it-qat-4bit"
    GEMMA_3_4B_IT_QAT_4BIT = "gemma-3-4b-it-qat-4bit"
    LLAMA_3_1_8B_INSTRUCT_4BIT = "llama-3.1-8b-instruct-4bit"
    LLAMA_3_2_1B_INSTRUCT_4BIT = "llama-3.2-1b-instruct-4bit"
    LLAMA_3_2_3B_INSTRUCT_4BIT = "llama-3.2-3b-instruct-4bit"
    MISTRAL_NEMO_INSTRUCT_2407_4BIT = "mistral-nemo-instruct-2407-4bit"
    QWEN2_5_7B_INSTRUCT_4BIT = "qwen2.5-7b-instruct-4bit"
    QWEN2_5_14B_INSTRUCT_4BIT = "qwen2.5-14b-instruct-4bit"
    QWEN2_5_CODER_14B_INSTRUCT_4BIT = "qwen2.5-coder-14b-instruct-4bit"
    QWEN3_0_6B_4BIT = "qwen3-0.6b-4bit"
    QWEN3_1_7B_3BIT = "qwen3-1.7b-3bit"
    QWEN3_4B_3BIT = "qwen3-4b-3bit"
    QWEN3_8B_3BIT = "qwen3-8b-3bit"

    @property
    def key(self) -> str:
        return self.value

    @property
    def model_path(self) -> str:
        return f"mlx-community/{self.value.replace('.', '.').replace('_', '-')}"


# Model key types
ModelKey = Literal[
    "dolphin3.0-llama3.1-8b-4bit",
    "gemma-3-1b-it-qat-4bit",
    "gemma-3-4b-it-qat-4bit",
    "llama-3.1-8b-instruct-4bit",
    "llama-3.2-1b-instruct-4bit",
    "llama-3.2-3b-instruct-4bit",
    "mistral-nemo-instruct-2407-4bit",
    "qwen2.5-7b-instruct-4bit",
    "qwen2.5-14b-instruct-4bit",
    "qwen2.5-coder-14b-instruct-4bit",
    "qwen3-0.6b-4bit",
    "qwen3-1.7b-3bit",
    "qwen3-4b-3bit",
    "qwen3-8b-3bit"
]

# Model value types
ModelValue = Literal[
    "mlx-community/Dolphin3.0-Llama3.1-8B-4bit",
    "mlx-community/gemma-3-1b-it-qat-4bit",
    "mlx-community/gemma-3-4b-it-qat-4bit",
    "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Qwen2.5-14B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
    "mlx-community/Qwen3-0.6B-4bit",
    "mlx-community/Qwen3-1.7B-3bit",
    "mlx-community/Qwen3-4B-3bit",
    "mlx-community/Qwen3-8B-3bit"
]

# Combined model type
ModelType = Union[ModelKey, ModelValue, ModelTypeEnum]


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
    "all-minilm-l6-v2-bf16",
    "all-minilm-l6-v2-8bit",
    "all-minilm-l6-v2-6bit",
    "all-minilm-l6-v2-4bit"
]

# Embed model value types
EmbedModelValue = Literal[
    "mlx-community/all-MiniLM-L6-v2-bf16",
    "mlx-community/all-MiniLM-L6-v2-8bit",
    "mlx-community/all-MiniLM-L6-v2-6bit",
    "mlx-community/all-MiniLM-L6-v2-4bit"
]

# Combined embed model type
EmbedModelType = Union[EmbedModelKey, EmbedModelValue, EmbedModelTypeEnum]
