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
    "bge-large",
    "pythia-70m",
    "qwen3-embedding-0.6b",
    "qwen3-reranker-0.6b",
    "snowflake-arctic-embed-m",
    "snowflake-arctic-embed,137m",
    "snowflake-arctic-embed-s",
    "specter",
    "bert-base-cased",
    "bert-base-multilingual-cased",
    "bert-base-uncased",
    "ms-marco-minilm-l-6-v2",
    "deepseek-r1",
    "deepseek-r1-distill-qwen-1.5b",
    "distilbert-base-uncased",
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "gpt2",
    "granite-embedding,278m",
    "granite-embedding",
    "e5-base-v2",
    "llama-3.1-8b",
    "llama-3.2-1b",
    "llama-3.2-3b",
    "deberta-v3-small",
    "mistral-7b-instruct-v0.3",
    "mxbai-embed-large",
    "deepseek-r1-distill-qwen-14b-4bit",
    "dolphin3.0-llama3.1-8b-4bit",
    "llama-3.1-8b-instruct-4bit",
    "llama-3.2-1b-instruct-4bit",
    "llama-3.2-3b-instruct-4bit",
    "mistral-nemo-instruct-2407-4bit",
    "qwen1.5-0.5b-chat-4bit",
    "qwen2.5-14b-instruct-4bit",
    "qwen2.5-7b-instruct-4bit",
    "qwen2.5-coder-14b-instruct-4bit",
    "qwen3-0.6b-4bit",
    "qwen3-0.6b-4bit-dwq-053125",
    "qwen3-1.7b-3bit",
    "qwen3-1.7b-4bit",
    "qwen3-1.7b-4bit-dwq-053125",
    "qwen3-4b-3bit",
    "qwen3-4b-4bit",
    "qwen3-8b-4bit",
    "qwen3-embedding-0.6b-4bit",
    "qwen3-embedding-4b-4bit-dwq",
    "all-minilm-l6-v2-4bit",
    "all-minilm-l6-v2-6bit",
    "all-minilm-l6-v2-8bit",
    "all-minilm-l6-v2-bf16",
    "dolphin3.0-llama3.2-3b-4bit",
    "gemma-3-12b-it-qat-4bit",
    "gemma-3-1b-it-qat-4bit",
    "gemma-3-4b-it-qat-4bit",
    "nomic-bert-2048",
    "nomic-embed-text",
    "roberta-base",
    "roberta-large",
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "all-mpnet-base-v2",
    "allenai-specter",
    "distilbert-base-nli-stsb-mean-tokens",
    "paraphrase-multilingual",
    "multi-qa-MiniLM-L6-cos-v1",
    "static-retrieval-mrl-en-v1",
]

# Model value types
LLMModelValue = Literal[
    "BAAI/bge-large-en-v1.5",
    "EleutherAI/pythia-70m",
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Reranker-0.6B",
    "Snowflake/snowflake-arctic-embed-m",
    "Snowflake/snowflake-arctic-embed-m-long",
    "Snowflake/snowflake-arctic-embed-s",
    "allenai/specter",
    "bert-base-cased",
    "bert-base-multilingual-cased",
    "bert-base-uncased",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    "cross-encoder/ms-marco-MiniLM-L12-v2",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "distilbert-base-uncased",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "gpt2",
    "ibm-granite/granite-embedding-278m-multilingual",
    "ibm-granite/granite-embedding-30m-english",
    "intfloat/e5-base-v2",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "microsoft/deberta-v3-small",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mixedbread-ai/mxbai-embed-large-v1",
    "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
    "mlx-community/Dolphin3.0-Llama3.1-8B-4bit",
    "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
    "mlx-community/Qwen1.5-0.5B-Chat-4bit",
    "mlx-community/Qwen2.5-14B-Instruct-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
    "mlx-community/Qwen3-0.6B-4bit",
    "mlx-community/Qwen3-0.6B-4bit-DWQ-053125",
    "mlx-community/Qwen3-1.7B-3bit",
    "mlx-community/Qwen3-1.7B-4bit-DWQ-053125",
    "mlx-community/Qwen3-4B-3bit",
    "mlx-community/Qwen3-4B-4bit-DWQ-053125",
    "mlx-community/Qwen3-8B-4bit-DWQ-053125",
    "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
    "mlx-community/all-MiniLM-L6-v2-4bit",
    "mlx-community/all-MiniLM-L6-v2-6bit",
    "mlx-community/all-MiniLM-L6-v2-8bit",
    "mlx-community/all-MiniLM-L6-v2-bf16",
    "mlx-community/dolphin3.0-llama3.2-3B-4Bit",
    "mlx-community/gemma-3-12b-it-qat-4bit",
    "mlx-community/gemma-3-1b-it-qat-4bit",
    "mlx-community/gemma-3-4b-it-qat-4bit",
    "nomic-ai/nomic-bert-2048",
    "nomic-ai/nomic-embed-text-v1.5",
    "roberta-base",
    "roberta-large",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/allenai-specter",
    "sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
    "sentence-transformers/paraphrase-MiniLM-L12-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "sentence-transformers/static-retrieval-mrl-en-v1",
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
    "bge-large",
    "pythia-70m",
    "qwen3-embedding-0.6b",
    "qwen3-reranker-0.6b",
    "snowflake-arctic-embed-m",
    "snowflake-arctic-embed,137m",
    "snowflake-arctic-embed-s",
    "specter",
    "bert-base-cased",
    "bert-base-multilingual-cased",
    "bert-base-uncased",
    "ms-marco-minilm-l-6-v2",
    "deepseek-r1",
    "deepseek-r1-distill-qwen-1.5b",
    "distilbert-base-uncased",
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "gpt2",
    "granite-embedding,278m",
    "granite-embedding",
    "e5-base-v2",
    "llama-3.1-8b",
    "llama-3.2-1b",
    "llama-3.2-3b",
    "deberta-v3-small",
    "mistral-7b-instruct-v0.3",
    "mxbai-embed-large",
    "deepseek-r1-distill-qwen-14b-4bit",
    "dolphin3.0-llama3.1-8b-4bit",
    "llama-3.1-8b-instruct-4bit",
    "llama-3.2-1b-instruct-4bit",
    "llama-3.2-3b-instruct-4bit",
    "mistral-nemo-instruct-2407-4bit",
    "qwen1.5-0.5b-chat-4bit",
    "qwen2.5-14b-instruct-4bit",
    "qwen2.5-7b-instruct-4bit",
    "qwen2.5-coder-14b-instruct-4bit",
    "qwen3-0.6b-4bit",
    "qwen3-0.6b-4bit-dwq-053125",
    "qwen3-1.7b-3bit",
    "qwen3-1.7b-4bit-dwq-053125",
    "qwen3-4b-3bit",
    "qwen3-4b-4bit",
    "qwen3-8b-4bit",
    "qwen3-embedding-0.6b-4bit",
    "qwen3-embedding-4b-4bit-dwq",
    "all-minilm-l6-v2-4bit",
    "all-minilm-l6-v2-6bit",
    "all-minilm-l6-v2-8bit",
    "all-minilm-l6-v2-bf16",
    "dolphin3.0-llama3.2-3b-4bit",
    "gemma-3-12b-it-qat-4bit",
    "gemma-3-1b-it-qat-4bit",
    "gemma-3-4b-it-qat-4bit",
    "nomic-bert-2048",
    "nomic-embed-text",
    "roberta-base",
    "roberta-large",
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "allenai-specter",
    "distilbert-base-nli-stsb-mean-tokens",
    "paraphrase-multilingual",
    "multi-qa-MiniLM-L6-cos-v1",
    "static-retrieval-mrl-en-v1",
    "ms-marco-MiniLM-L-6-v2",
    "ms-marco-MiniLM-L6-v2",
    "ms-marco-MiniLM-L-12-v2",
    "ms-marco-MiniLM-L12-v2",
]
EmbedModelKey = Union[EmbedModelKey, LLMModelKey]

# Embed model value types
EmbedModelValue = Literal[
    "BAAI/bge-large-en-v1.5",
    "EleutherAI/pythia-70m",
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Reranker-0.6B",
    "Snowflake/snowflake-arctic-embed-m",
    "Snowflake/snowflake-arctic-embed-m-long",
    "Snowflake/snowflake-arctic-embed-s",
    "allenai/specter",
    "bert-base-cased",
    "bert-base-multilingual-cased",
    "bert-base-uncased",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    "cross-encoder/ms-marco-MiniLM-L12-v2",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "distilbert-base-uncased",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "gpt2",
    "ibm-granite/granite-embedding-278m-multilingual",
    "ibm-granite/granite-embedding-30m-english",
    "intfloat/e5-base-v2",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "microsoft/deberta-v3-small",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mixedbread-ai/mxbai-embed-large-v1",
    "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
    "mlx-community/Dolphin3.0-Llama3.1-8B-4bit",
    "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
    "mlx-community/Qwen1.5-0.5B-Chat-4bit",
    "mlx-community/Qwen2.5-14B-Instruct-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
    "mlx-community/Qwen3-0.6B-4bit",
    "mlx-community/Qwen3-0.6B-4bit-DWQ-053125",
    "mlx-community/Qwen3-1.7B-3bit",
    "mlx-community/Qwen3-1.7B-4bit-DWQ-053125",
    "mlx-community/Qwen3-4B-3bit",
    "mlx-community/Qwen3-4B-4bit-DWQ-053125",
    "mlx-community/Qwen3-8B-4bit-DWQ-053125",
    "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
    "mlx-community/all-MiniLM-L6-v2-4bit",
    "mlx-community/all-MiniLM-L6-v2-6bit",
    "mlx-community/all-MiniLM-L6-v2-8bit",
    "mlx-community/all-MiniLM-L6-v2-bf16",
    "mlx-community/dolphin3.0-llama3.2-3B-4Bit",
    "mlx-community/gemma-3-12b-it-qat-4bit",
    "mlx-community/gemma-3-1b-it-qat-4bit",
    "mlx-community/gemma-3-4b-it-qat-4bit",
    "nomic-ai/nomic-bert-2048",
    "nomic-ai/nomic-embed-text-v1.5",
    "roberta-base",
    "roberta-large",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/allenai-specter",
    "sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
    "sentence-transformers/paraphrase-MiniLM-L12-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "sentence-transformers/static-retrieval-mrl-en-v1",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "cross-encoder/ms-marco-MiniLM-L12-v2",
]
EmbedModelValue = Union[EmbedModelValue, LLMModelValue]

# Combined llm model type
# LLMModelType = Union[LLMModelKey, LLMModelValue]
LLMModelType = Union[LLMModelKey, LLMModelValue]

# Combined embed model type
EmbedModelType = Union[EmbedModelKey, EmbedModelValue]

ModelKey = Union[LLMModelKey, EmbedModelKey]
ModelValue = Union[LLMModelValue, EmbedModelValue]
ModelType = Union[LLMModelType, EmbedModelType]


# Extract Enum classes from the union
enum_types_in_model_type = [t for t in get_args(
    ModelType) if isinstance(t, type) and issubclass(t, Enum)]


def get_model_keys() -> List[str]:
    model_keys: List[str] = []
    for model_type in get_args(ModelKey):
        model_keys.extend(get_args(model_type))
    return model_keys


def get_model_values() -> List[str]:
    model_values: List[str] = []
    for model_type in get_args(ModelValue):
        model_values.extend(get_args(model_type))
    return model_values


def get_model_types() -> List[str]:
    model_types: List[str] = []
    # Add keys and values
    for model_type in get_args(ModelType):
        model_types.extend(get_args(model_type))
    # Add Enum values
    for enum_type in enum_types_in_model_type:
        model_types.extend(e.value for e in enum_type)
    return model_types


# All model keys from LLMModelKey + EmbedModelKey
model_keys_list: List[str] = get_model_keys()

# All model values from LLMModelValue + EmbedModelValue
model_values_list: List[str] = get_model_values()

# All model types: keys + values + Enum values
model_types_list: List[str] = get_model_types()


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
