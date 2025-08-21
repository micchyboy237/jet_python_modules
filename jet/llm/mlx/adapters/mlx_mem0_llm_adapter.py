from typing import Any, Dict, List, Optional, Union, Literal
from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase
from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import ChatTemplateArgs
from jet.models.model_types import LLMModelType
from jet.models.config import DEFAULT_MODEL


class MLXMem0LLMConfig(BaseLlmConfig):
    """
    Configuration class for MLXMem0LLMAdapter.
    Extends BaseLlmConfig with MLX-specific parameters.
    """

    def __init__(
        self,
        model: LLMModelType = DEFAULT_MODEL,
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        max_tokens: int = 2000,
        top_p: float = 0.1,
        top_k: int = 1,
        enable_vision: bool = False,
        vision_details: Optional[str] = "auto",
        http_client_proxies: Optional[Union[Dict, str]] = None,
        adapter_path: Optional[str] = None,
        draft_model: Optional[str] = None,
        trust_remote_code: bool = False,
        chat_template: Optional[str] = None,
        use_default_chat_template: bool = True,
        chat_template_args: Optional[ChatTemplateArgs] = None,
        seed: Optional[int] = None,
        device: Optional[Literal["cpu", "mps"]] = "mps",
        log_dir: Optional[str] = None,
        prompt_cache: Optional[List[Any]] = None
    ):
        """
        Initialize the MLXMem0LLMConfig with MLX-specific parameters.

        Args:
            model: The model identifier to use
            temperature: Controls randomness of output (0.0 to 2.0)
            api_key: API key for the LLM provider (optional for MLX)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            top_k: Top-k sampling parameter
            enable_vision: Whether to enable vision capabilities
            vision_details: Level of detail for vision processing ("low", "high", "auto")
            http_client_proxies: Proxy settings for HTTP client
            adapter_path: Path to model adapter (optional)
            draft_model: Draft model for speculative decoding (optional)
            trust_remote_code: Whether to trust remote code
            chat_template: Custom chat template (optional)
            use_default_chat_template: Whether to use default chat template
            chat_template_args: Additional chat template arguments
            seed: Random seed for reproducibility
            device: Device to run the model ("cpu" or "mps")
            log_dir: Directory for logging
            prompt_cache: Cache for prompt processing
        """
        super().__init__(
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            enable_vision=enable_vision,
            vision_details=vision_details,
            http_client_proxies=http_client_proxies
        )
        self.adapter_path = adapter_path
        self.draft_model = draft_model
        self.trust_remote_code = trust_remote_code
        self.chat_template = chat_template
        self.use_default_chat_template = use_default_chat_template
        self.chat_template_args = chat_template_args
        self.seed = seed
        self.device = device
        self.log_dir = log_dir
        self.prompt_cache = prompt_cache


class MLXMem0LLMAdapter(LLMBase):
    """
    Adapter class for integrating MLX with Mem0 LLM framework.
    Extends LLMBase to provide MLX-specific response generation.
    """

    def __init__(self, config: Optional[Union[MLXMem0LLMConfig, Dict]] = None):
        """
        Initialize the MLXMem0LLMAdapter with configuration.
        Args:
            config: Configuration for the MLX model, defaults to None.
                   Can be an MLXMem0LLMConfig instance or a dictionary.
        """
        if config is None:
            self.config = MLXMem0LLMConfig()
        elif isinstance(config, dict):
            self.config = MLXMem0LLMConfig(**config)
        else:
            self.config = config
        super().__init__(self.config)
        self.mlx_client = MLX(
            model=self.config.model,
            adapter_path=self.config.adapter_path,
            draft_model=self.config.draft_model,
            trust_remote_code=self.config.trust_remote_code,
            chat_template=self.config.chat_template,
            use_default_chat_template=self.config.use_default_chat_template,
            chat_template_args=self.config.chat_template_args,
            seed=self.config.seed,
            device=self.config.device,
            log_dir=self.config.log_dir,
            prompt_cache=self.config.prompt_cache,
            with_history=False
        )

    def _validate_config(self):
        """
        Validate the MLX-specific configuration.
        """
        super()._validate_config()
        if not self.config.model:
            raise ValueError(
                "MLX model name must be specified in the configuration")

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs
    ) -> Union[str, Dict]:
        """
        Generate a response using the MLX client.
        Args:
            messages: List of message dictionaries containing 'role' and 'content'.
            tools: Optional list of tools the model can call.
            tool_choice: Tool choice method, defaults to "auto".
            **kwargs: Additional MLX-specific parameters.
        Returns:
            Union[str, Dict]: The generated response, either as a string or dictionary.
        """
        supported_params = self._get_supported_params(
            messages=messages,
            tools=tools,
            # tool_choice=tool_choice,
            **kwargs
        )
        # Use kwargs if provided, otherwise fall back to config values
        supported_params.update({
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k)
        })
        system_prompt = None
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content")
            else:
                filtered_messages.append(msg)
        generation_settings = {
            "messages": filtered_messages,
            "system_prompt": system_prompt,
            "tools": tools,
            # "tool_choice": tool_choice,
            **supported_params
        }
        generation_settings.pop("response_format")
        response = self.mlx_client.chat(**generation_settings)
        if isinstance(response, dict) and response.get("choices"):
            return response["choices"][0].get("message", {}).get("content", "")
        return str(response)
