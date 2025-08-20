from abc import ABC
from typing import Any, Optional, Dict, Literal, List, Iterator, TypedDict, Union
from threading import Lock
from jet.llm.mlx.config import DEFAULT_MODEL
from jet.llm.mlx.mlx_types import ChatTemplateArgs
from jet.logger import logger
from pathlib import Path
from jet.models.model_registry.base import BaseModelRegistry
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoConfig, PretrainedConfig
from tokenizers import Tokenizer
from jet.llm.mlx.base import MLX
from jet.llm.mlx.client import CompletionResponse, Message
from jet.models.model_types import LLMModelType, RoleMapping, Tool
from jet.models.utils import resolve_model_value, get_local_repo_dir
from jet.db.postgres.config import DEFAULT_HOST, DEFAULT_PASSWORD, DEFAULT_PORT, DEFAULT_USER
from jet.data.utils import generate_hash


class ModelFeatures(TypedDict):
    """Configuration options for loading models."""
    model: LLMModelType
    adapter_path: Optional[str]
    draft_model: Optional[LLMModelType]
    trust_remote_code: bool
    chat_template: Optional[str]
    use_default_chat_template: bool
    chat_template_args: Optional[ChatTemplateArgs]
    prompt_cache: Optional[List[Any]]
    dbname: Optional[str]
    user: Optional[str]
    password: Optional[str]
    host: Optional[str]
    port: Optional[str]
    session_id: Optional[str]
    with_history: bool
    seed: Optional[int]
    log_dir: Optional[str]
    device: Optional[Literal["cpu", "mps"]]


class MLXModelRegistry(BaseModelRegistry):
    """Abstract base class for MLX-based model registries."""
    _models: Dict[str, nn.Module] = {}
    _tokenizers: Dict[str, PreTrainedTokenizerBase] = {}
    _configs: Dict[str, PretrainedConfig] = {}
    _model_lock = Lock()  # Lock for thread-safe model caching
    _tokenizer_lock = Lock()  # Lock for thread-safe tokenizer caching
    _config_lock = Lock()  # Lock for thread-safe config caching

    @staticmethod
    def load_model(
        # Model Config
        model: LLMModelType = DEFAULT_MODEL,
        adapter_path: Optional[str] = None,
        draft_model: Optional[LLMModelType] = None,
        trust_remote_code: bool = False,
        chat_template: Optional[str] = None,
        use_default_chat_template: bool = True,
        chat_template_args: Optional[ChatTemplateArgs] = None,
        prompt_cache: Optional[List[Any]] = None,
        # DB Config
        dbname: Optional[str] = None,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASSWORD,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        overwrite_db: bool = False,
        session_id: Optional[str] = None,
        with_history: bool = False,
        seed: Optional[int] = None,
        log_dir: Optional[str] = None,
        device: Optional[Literal["cpu", "mps"]] = "mps"
    ) -> MLX:
        """Load or retrieve an MLX model statically."""
        # Generate cache key based on model-defining parameters
        cache_key = generate_hash(
            model=resolve_model_value(model),
            adapter_path=adapter_path,
            draft_model=resolve_model_value(
                draft_model) if draft_model else None,
            trust_remote_code=trust_remote_code,
            chat_template=chat_template,
            use_default_chat_template=use_default_chat_template,
            chat_template_args=chat_template_args,
            prompt_cache=prompt_cache,
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
            overwrite_db=overwrite_db,
            device=device
        )
        instance = MLXModelRegistry()
        with instance._model_lock:
            if cache_key in instance._models:
                logger.info(
                    f"Reusing existing MLX model for cache_key: {cache_key}")
                return instance._models[cache_key]

        logger.info(f"Loading MLX model for cache_key: {cache_key}")
        try:
            model_instance = instance._load_mlx_model(
                model=model,
                adapter_path=adapter_path,
                draft_model=draft_model,
                trust_remote_code=trust_remote_code,
                chat_template=chat_template,
                use_default_chat_template=use_default_chat_template,
                chat_template_args=chat_template_args,
                prompt_cache=prompt_cache,
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port,
                overwrite_db=overwrite_db,
                session_id=session_id,
                with_history=with_history,
                seed=seed,
                log_dir=log_dir,
                device=device,
            )
            with instance._model_lock:
                instance._models[cache_key] = model_instance
            return model_instance
        except Exception as e:
            logger.error(
                f"Failed to load MLX model for cache_key {cache_key}: {str(e)}")
            raise ValueError(
                f"Could not load MLX model for cache_key {cache_key}: {str(e)}")

    @staticmethod
    def get_tokenizer(model_id: LLMModelType) -> Optional[PreTrainedTokenizerBase]:
        """Load or retrieve a tokenizer for the MLX model."""
        resolved_model_id = resolve_model_value(model_id)
        cache_key = generate_hash(model_id=resolved_model_id)
        with MLXModelRegistry._tokenizer_lock:
            if cache_key in MLXModelRegistry._tokenizers:
                logger.info(f"Reusing tokenizer for cache_key: {cache_key}")
                return MLXModelRegistry._tokenizers[cache_key]

        logger.info(f"Loading tokenizer for cache_key: {cache_key}")
        try:
            cache_dir = Path(get_local_repo_dir(resolved_model_id))
            tokenizer = AutoTokenizer.from_pretrained(resolved_model_id)
            logger.info(
                f"Successfully loaded tokenizer for {resolved_model_id} from local cache: {cache_dir}")
            with MLXModelRegistry._tokenizer_lock:
                MLXModelRegistry._tokenizers[cache_key] = tokenizer
            return tokenizer
        except Exception as e:
            logger.error(
                f"Failed to load tokenizer for cache_key {cache_key}: {str(e)}")
            raise ValueError(
                f"Could not load tokenizer for cache_key {cache_key}: {str(e)}")

    @staticmethod
    def get_config(model_id: LLMModelType) -> Optional[PretrainedConfig]:
        """Load or retrieve a config for the MLX model."""
        resolved_model_id = resolve_model_value(model_id)
        cache_key = generate_hash(model_id=resolved_model_id)
        with MLXModelRegistry._config_lock:
            if cache_key in MLXModelRegistry._configs:
                logger.info(f"Reusing config for cache_key: {cache_key}")
                return MLXModelRegistry._configs[cache_key]

        logger.info(f"Loading config for cache_key: {cache_key}")
        try:
            cache_dir = Path(get_local_repo_dir(resolved_model_id))
            snapshot_dir = cache_dir / "snapshots"
            config_files = list(snapshot_dir.glob("*/config.json"))
            logger.debug(
                f"Found {len(config_files)} config.json files in {snapshot_dir}")

            for config_path in config_files:
                try:
                    resolved_path = config_path.resolve()
                    logger.debug(f"Resolved {config_path} to {resolved_path}")
                    if not resolved_path.is_file():
                        logger.warning(
                            f"Resolved path is not a file: {resolved_path}")
                        continue
                    config = AutoConfig.from_pretrained(str(resolved_path))
                    logger.info(
                        f"Successfully loaded config from local cache: {resolved_path}")
                    with MLXModelRegistry._config_lock:
                        MLXModelRegistry._configs[cache_key] = config
                    return config
                except Exception as local_e:
                    logger.error(
                        f"Failed to load config from {resolved_path}: {str(local_e)}")
                    continue

            error_msg = f"Could not load config for cache_key {cache_key} from local cache."
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            logger.error(
                f"Failed to load config for cache_key {cache_key}: {str(e)}")
            raise ValueError(
                f"Could not load config for cache_key {cache_key}: {str(e)}")

    def clear(self) -> None:
        """Clear all cached models, tokenizers, and configs."""
        with self._model_lock:
            self._models.clear()
        with self._tokenizer_lock:
            self._tokenizers.clear()
        with self._config_lock:
            self._configs.clear()
        logger.info("MLX registry cleared")

    def _load_mlx_model(self, model: LLMModelType, device: Optional[Literal["cpu", "mps"]], *args, **kwargs) -> MLX:
        """Load an MLX model with specified device, precision, and generation features."""
        try:
            _model = MLX(model=model, device=device, *args, **kwargs)
            # Reset model prompt cache
            _model.reset_model()
            logger.info(
                f"Successfully loaded MLX model {model} on device {device}")
            return _model
        except Exception as e:
            logger.error(f"Failed to load MLX model {model}: {str(e)}")
            raise ValueError(f"Could not load MLX model {model}: {str(e)}")
