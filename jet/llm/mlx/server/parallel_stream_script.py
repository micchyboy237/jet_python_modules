import json
import uuid
from typing import List, Optional, Union, Dict
from jet.llm.mlx.base import MLX
from jet.logger import logger
from mpi4py import MPI
from jet.llm.mlx.mlx_types import Message, ModelType, RoleMapping, Tool, CompletionResponse
from jet.llm.mlx.models import AVAILABLE_MODELS, get_model_limits

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def parallel_stream_generate(
    model: ModelType,
    prompts: List[str],
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: int = 20,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    logit_bias: Optional[Dict[int, float]] = None,
    logprobs: int = -1,
    stop: Optional[Union[str, List[str]]] = None,
    verbose: bool = False,
    worker_verbose: bool = False,
    task_id: str = "",
    stream: bool = True
) -> None:
    """
    Parallel text generation across MPI workers, supporting both streaming and non-streaming modes.
    Sends chunks or results to rank 0 via MPI.
    """
    if rank != 0:
        try:
            # Validate model path
            if model not in AVAILABLE_MODELS.values():
                raise ValueError(
                    f"Invalid model path: {model}. Must be one of {list(AVAILABLE_MODELS.values())}")
            if verbose:
                logger.info(f"Rank {rank}: Validating model path {model}")

            # Check model limits
            max_context, max_embeddings = get_model_limits(model)
            if not max_context or not max_embeddings:
                raise ValueError(
                    f"Invalid model configuration for {model}: max_context={max_context}, max_embeddings={max_embeddings}")
            if verbose:
                logger.info(
                    f"Rank {rank}: Model {model} limits: max_context={max_context}, max_embeddings={max_embeddings}")

            # Initialize MLX
            mlx = MLX(model=model)
        except Exception as e:
            error_msg = f"Rank {rank}: Failed to load model {model}: {str(e)}"
            logger.error(error_msg)
            comm.send({
                "type": "error",
                "message": error_msg,
                "prompt_id": None,
                "task_id": task_id
            }, dest=0)
            return

        for prompt in prompts:
            prompt_id = str(uuid.uuid4())
            try:
                if not isinstance(prompt, str):
                    raise ValueError(
                        f"Prompt must be a string, got {type(prompt)}")
                if worker_verbose:
                    logger.info(
                        f"Rank {rank}: Generating for prompt: {prompt[:50]}...")
                tokens_generated = 0
                if stream:
                    for chunk in mlx.stream_generate(
                        prompt=prompt,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        repetition_context_size=repetition_context_size,
                        xtc_probability=xtc_probability,
                        xtc_threshold=xtc_threshold,
                        logit_bias=logit_bias,
                        logprobs=logprobs,
                        stop=stop
                    ):
                        content = chunk["choices"][0].get("text", "")
                        tokens_generated += len(content.split())
                        comm.send({
                            "type": "chunk",
                            "prompt": prompt,
                            "content": content,
                            "prompt_id": prompt_id,
                            "task_id": task_id
                        }, dest=0)
                        if worker_verbose:
                            logger.info(
                                f"Rank {rank}: Prompt ID {prompt_id}: {content[:50]}...")
                else:
                    response = mlx.generate(
                        prompt=prompt,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        repetition_context_size=repetition_context_size,
                        xtc_probability=xtc_probability,
                        xtc_threshold=xtc_threshold,
                        logit_bias=logit_bias,
                        logprobs=logprobs,
                        stop=stop
                    )
                    content = response["choices"][0].get("text", "")
                    tokens_generated += len(content.split())
                    comm.send({
                        "type": "result",
                        "prompt": prompt,
                        "content": content,
                        "prompt_id": prompt_id,
                        "task_id": task_id,
                        "truncated": tokens_generated >= max_tokens
                    }, dest=0)

                comm.send({
                    "type": "result",
                    "prompt": prompt,
                    "content": "",
                    "prompt_id": prompt_id,
                    "task_id": task_id,
                    "truncated": tokens_generated >= max_tokens
                }, dest=0)
            except Exception as e:
                error_msg = f"Rank {rank}: Prompt ID {prompt_id} failed: {str(e)}"
                if worker_verbose:
                    logger.error(error_msg)
                comm.send({
                    "type": "error",
                    "message": error_msg,
                    "prompt_id": prompt_id,
                    "task_id": task_id
                }, dest=0)
    else:
        pass


def parallel_chat_generate(
    model: ModelType,
    messages: List[str],
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: int = 20,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    logit_bias: Optional[Dict[int, float]] = None,
    logprobs: int = -1,
    stop: Optional[Union[str, List[str]]] = None,
    role_mapping: Optional[RoleMapping] = None,
    tools: Optional[List[Tool]] = None,
    system_prompt: Optional[str] = None,
    verbose: bool = False,
    worker_verbose: bool = False,
    task_id: str = "",
    session_id: Optional[str] = None,
    stream: bool = True
) -> None:
    """
    Parallel chat generation across MPI workers, supporting both streaming and non-streaming modes.
    Sends chunks or results to rank 0 via MPI, maintaining chat history with session_id.
    """
    if rank != 0:
        try:
            # Validate model path
            if model not in AVAILABLE_MODELS.values():
                raise ValueError(
                    f"Invalid model path: {model}. Must be one of {list(AVAILABLE_MODELS.values())}")
            if verbose:
                logger.info(f"Rank {rank}: Validating model path {model}")

            # Check model limits
            max_context, max_embeddings = get_model_limits(model)
            if not max_context or not max_embeddings:
                raise ValueError(
                    f"Invalid model configuration for {model}: max_context={max_context}, max_embeddings={max_embeddings}")
            if verbose:
                logger.info(
                    f"Rank {rank}: Model {model} limits: max_context={max_context}, max_embeddings={max_embeddings}")

            # Initialize MLX
            mlx = MLX(model=model, session_id=session_id)
        except Exception as e:
            error_msg = f"Rank {rank}: Failed to load model {model}: {str(e)}"
            logger.error(error_msg)
            comm.send({
                "type": "error",
                "message": error_msg,
                "prompt_id": None,
                "task_id": task_id
            }, dest=0)
            return

        for message_str in messages:
            prompt_id = str(uuid.uuid4())
            try:
                message_list = json.loads(message_str)
                if not isinstance(message_list, list) or not all(isinstance(m, dict) and "role" in m and "content" in m for m in message_list):
                    raise ValueError(
                        f"Messages must be a list of dictionaries with 'role' and 'content' keys, got {message_str}")
                messages_typed: List[Message] = message_list
                if worker_verbose:
                    logger.info(
                        f"Rank {rank}: Generating for messages: {messages_typed}")
                tokens_generated = 0
                if stream:
                    for chunk in mlx.stream_chat(
                        messages=messages_typed,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        repetition_context_size=repetition_context_size,
                        xtc_probability=xtc_probability,
                        xtc_threshold=xtc_threshold,
                        logit_bias=logit_bias,
                        logprobs=logprobs,
                        stop=stop,
                        role_mapping=role_mapping,
                        tools=tools,
                        system_prompt=system_prompt
                    ):
                        content = chunk["choices"][0].get(
                            "message", {}).get("content", "")
                        tokens_generated += len(content.split())
                        comm.send({
                            "type": "chunk",
                            "prompt": message_str,
                            "content": content,
                            "prompt_id": prompt_id,
                            "task_id": task_id
                        }, dest=0)
                        if worker_verbose:
                            logger.info(
                                f"Rank {rank}: Prompt ID {prompt_id}: {content[:50]}...")
                else:
                    response = mlx.chat(
                        messages=messages_typed,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        repetition_context_size=repetition_context_size,
                        xtc_probability=xtc_probability,
                        xtc_threshold=xtc_threshold,
                        logit_bias=logit_bias,
                        logprobs=logprobs,
                        stop=stop,
                        role_mapping=role_mapping,
                        tools=tools,
                        system_prompt=system_prompt
                    )
                    content = response["choices"][0].get(
                        "message", {}).get("content", "")
                    tokens_generated += len(content.split())
                    comm.send({
                        "type": "result",
                        "prompt": message_str,
                        "content": content,
                        "prompt_id": prompt_id,
                        "task_id": task_id,
                        "truncated": tokens_generated >= max_tokens
                    }, dest=0)

                comm.send({
                    "type": "result",
                    "prompt": message_str,
                    "content": "",
                    "prompt_id": prompt_id,
                    "task_id": task_id,
                    "truncated": tokens_generated >= max_tokens
                }, dest=0)
            except Exception as e:
                error_msg = f"Rank {rank}: Prompt ID {prompt_id} failed: {str(e)}"
                if worker_verbose:
                    logger.error(error_msg)
                comm.send({
                    "type": "error",
                    "message": error_msg,
                    "prompt_id": prompt_id,
                    "task_id": task_id
                }, dest=0)
    else:
        pass
