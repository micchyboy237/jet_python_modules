from abc import ABC
from typing import Optional, Dict, Literal, List, Iterator, TypedDict, Union
from threading import Lock
import uuid
from jet.data.utils import generate_hash, generate_unique_hash
from jet.llm.mlx.config import DEFAULT_MODEL
from jet.logger import logger
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoConfig, PretrainedConfig
from tokenizers import Tokenizer
from jet.llm.mlx.base import MLX
from jet.llm.mlx.client import CompletionResponse, Message
from jet.models.model_types import LLMModelType, RoleMapping, Tool
from jet.models.utils import resolve_model_value, get_local_repo_dir
from jet.db.postgres.config import DEFAULT_HOST, DEFAULT_PASSWORD, DEFAULT_PORT, DEFAULT_USER
from .base import TransformersModelRegistry


class ModelFeatures(TypedDict):
    """Configuration options for loading models."""
    model: LLMModelType
    adapter_path: Optional[str]
    draft_model: Optional[LLMModelType]
    trust_remote_code: bool
    chat_template: Optional[str]
    use_default_chat_template: bool
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


class MLXModelRegistry(TransformersModelRegistry):
    """Abstract base class for MLX-based model registries."""
    _models: Dict[str, nn.Module] = {}
    _tokenizers: Dict[str, PreTrainedTokenizerBase] = {}
    _configs: Dict[str, PretrainedConfig] = {}
    _model_lock = Lock()  # Lock for thread-safe model caching
    _tokenizer_lock = Lock()  # Lock for thread-safe tokenizer caching
    _config_lock = Lock()  # Lock for thread-safe config caching

    def __init__(self):
        """Initialize the registry with a unique session ID."""
        self.session_id = generate_unique_hash()

    @staticmethod
    def load_model(
        # Model Config
        model: LLMModelType = DEFAULT_MODEL,
        adapter_path: Optional[str] = None,
        draft_model: Optional[LLMModelType] = None,
        trust_remote_code: bool = False,
        chat_template: Optional[str] = None,
        use_default_chat_template: bool = True,
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
        # Get the Singleton instance
        instance = MLXModelRegistry()
        resolved_model_id = resolve_model_value(model)
        # Use instance session_id if none provided
        effective_session_id = session_id if session_id is not None else instance.session_id
        cache_key = generate_hash(
            model=resolved_model_id,
            adapter_path=adapter_path,
            draft_model=draft_model,
            trust_remote_code=trust_remote_code,
            chat_template=chat_template,
            use_default_chat_template=use_default_chat_template,
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
            overwrite_db=overwrite_db,
            session_id=effective_session_id,
            with_history=with_history,
            seed=seed,
            log_dir=log_dir,
            device=device
        )
        with instance._model_lock:
            if cache_key in instance._models:
                logger.info(
                    f"Reusing existing MLX model for cache_key: {cache_key}")
                return instance._models[cache_key]

        logger.info(f"Loading MLX model for cache_key: {cache_key}")
        try:
            model = instance._load_mlx_model(
                model=resolved_model_id,
                adapter_path=adapter_path,
                draft_model=draft_model,
                trust_remote_code=trust_remote_code,
                chat_template=chat_template,
                use_default_chat_template=use_default_chat_template,
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port,
                overwrite_db=overwrite_db,
                session_id=effective_session_id,
                with_history=with_history,
                seed=seed,
                log_dir=log_dir,
                device=device,
            )
            with instance._model_lock:
                instance._models[cache_key] = model
            return model
        except Exception as e:
            logger.error(
                f"Failed to load MLX model {cache_key}: {str(e)}")
            raise ValueError(
                f"Could not load MLX model {cache_key}: {str(e)}")

    @staticmethod
    def get_tokenizer(model_id: LLMModelType) -> Optional[PreTrainedTokenizerBase]:
        """Load or retrieve a tokenizer for the MLX model."""
        resolved_model_id = resolve_model_value(model_id)
        with MLXModelRegistry._tokenizer_lock:
            if resolved_model_id in MLXModelRegistry._tokenizers:
                logger.info(
                    f"Reusing tokenizer for model_id: {resolved_model_id}")
                return MLXModelRegistry._tokenizers[resolved_model_id]

        logger.info(f"Loading tokenizer for model_id: {resolved_model_id}")
        try:
            cache_dir = Path(get_local_repo_dir(resolved_model_id))
            tokenizer = AutoTokenizer.from_pretrained(resolved_model_id)
            logger.info(
                f"Successfully loaded tokenizer for {resolved_model_id} from local cache: {cache_dir}")
            with MLXModelRegistry._tokenizer_lock:
                MLXModelRegistry._tokenizers[resolved_model_id] = tokenizer
            return tokenizer
        except Exception as e:
            logger.error(
                f"Failed to load tokenizer for {resolved_model_id}: {str(e)}")
            raise ValueError(
                f"Could not load tokenizer for {resolved_model_id}: {str(e)}")

    @staticmethod
    def get_config(model_id: LLMModelType) -> Optional[PretrainedConfig]:
        """Load or retrieve a config for the MLX model."""
        resolved_model_id = resolve_model_value(model_id)
        with MLXModelRegistry._config_lock:
            if resolved_model_id in MLXModelRegistry._configs:
                logger.info(
                    f"Reusing config for model_id: {resolved_model_id}")
                return MLXModelRegistry._configs[resolved_model_id]

        logger.info(f"Loading config for model_id: {resolved_model_id}")
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
                        MLXModelRegistry._configs[resolved_model_id] = config
                    return config
                except Exception as local_e:
                    logger.error(
                        f"Failed to load config from {resolved_path}: {str(local_e)}")
                    continue

            error_msg = f"Could not load config for {resolved_model_id} from local cache."
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            logger.error(
                f"Failed to load config {resolved_model_id}: {str(e)}")
            raise ValueError(
                f"Could not load config {resolved_model_id}: {str(e)}")

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
            model = MLX(model=model, device=device, *args, **kwargs)
            logger.info(
                f"Successfully loaded MLX model {model} on device {device}")
            return model
        except Exception as e:
            logger.error(f"Failed to load MLX model {model}: {str(e)}")
            raise ValueError(f"Could not load MLX model {model}: {str(e)}")
