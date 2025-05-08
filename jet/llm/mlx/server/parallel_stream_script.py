import json
import uuid
from typing import List, Optional, Union, Dict
from jet.llm.mlx.base import MLX
from jet.logger import logger
from mpi4py import MPI
from jet.llm.mlx.mlx_types import Message, ModelType, RoleMapping, Tool, CompletionResponse
from jet.llm.mlx.models import AVAILABLE_MODELS, get_model_limits, resolve_model

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def parallel_stream_generate(
    model: ModelType,
    # Updated to expect list of {"prompt": ..., "prompt_id": ...}
    prompts: List[Dict[str, str]],
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
    if rank != 0:
        try:
            model = resolve_model(model, AVAILABLE_MODELS)
            if verbose:
                logger.info(f"Rank {rank}: Validating model path {model}")
            max_context, max_embeddings = get_model_limits(model)
            if not max_context or not max_embeddings:
                raise ValueError(
                    f"Invalid model configuration for {model}: max_context={max_context}, max_embeddings={max_embeddings}")
            if verbose:
                logger.info(
                    f"Rank {rank}: Model {model} limits: max_context={max_context}, max_embeddings={max_embeddings}")
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
        for prompt_data in prompts:
            prompt = prompt_data["prompt"]
            prompt_id = prompt_data["prompt_id"]
            try:
                if not isinstance(prompt, str):
                    raise ValueError(
                        f"Prompt must be a string, got {type(prompt)}")
                if not isinstance(prompt_id, str):
                    raise ValueError(
                        f"Prompt ID must be a string, got {type(prompt_id)}")
                if worker_verbose:
                    logger.info(
                        f"Rank {rank}: Generating for prompt: {prompt[:50]}... with prompt_id: {prompt_id}")
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
                        if worker_verbose:
                            logger.info(
                                f"Rank {rank}: Prompt ID {prompt_id}: Stream chunk content={content[:50]}...")
                        tokens_generated += len(content.split())
                        comm.send({
                            "type": "chunk",
                            "prompt": prompt,
                            "content": content,
                            "prompt_id": prompt_id,
                            "task_id": task_id
                        }, dest=0)
                    # Send final result for streaming
                    comm.send({
                        "type": "result",
                        "prompt": prompt,
                        "content": "",
                        "prompt_id": prompt_id,
                        "task_id": task_id,
                        "truncated": tokens_generated >= max_tokens
                    }, dest=0)
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
                    if worker_verbose:
                        logger.info(
                            f"Rank {rank}: Prompt ID {prompt_id}: Full response={response}")
                    content = response["choices"][0].get("text", "")
                    if not content and worker_verbose:
                        logger.warning(
                            f"Rank {rank}: Prompt ID {prompt_id}: Empty content in response")
                    tokens_generated += len(content.split())
                    comm.send({
                        "type": "result",
                        "prompt": prompt,
                        "content": content,
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
    # Updated to expect list of {"prompt": ..., "prompt_id": ...}
    messages: List[Dict[str, str]],
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
    if rank != 0:
        try:
            model = resolve_model(model, AVAILABLE_MODELS)
            if verbose:
                logger.info(f"Rank {rank}: Validating model path {model}")
            max_context, max_embeddings = get_model_limits(model)
            if not max_context or not max_embeddings:
                raise ValueError(
                    f"Invalid model configuration for {model}: max_context={max_context}, max_embeddings={max_embeddings}")
            if verbose:
                logger.info(
                    f"Rank {rank}: Model {model} limits: max_context={max_context}, max_embeddings={max_embeddings}")
            mlx = MLX(
                model=model, session_id=session_id if session_id else str(uuid.uuid4()))
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
        for message_data in messages:
            message_str = message_data["prompt"]
            prompt_id = message_data["prompt_id"]
            try:
                message_list = json.loads(message_str)
                if not isinstance(message_list, list) or not all(isinstance(m, dict) and "role" in m and "content" in m for m in message_list):
                    raise ValueError(
                        f"Messages must be a list of dictionaries with 'role' and 'content' keys, got {message_str}")
                if not isinstance(prompt_id, str):
                    raise ValueError(
                        f"Prompt ID must be a string, got {type(prompt_id)}")
                messages_typed: List[Message] = message_list
                if worker_verbose:
                    logger.info(
                        f"Rank {rank}: Generating for messages: {messages_typed} with prompt_id: {prompt_id}")
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
                        if worker_verbose:
                            logger.info(
                                f"Rank {rank}: Prompt ID {prompt_id}: Stream chunk content={content[:50]}...")
                        tokens_generated += len(content.split())
                        comm.send({
                            "type": "chunk",
                            "prompt": message_str,
                            "content": content,
                            "prompt_id": prompt_id,
                            "task_id": task_id
                        }, dest=0)
                    # Send final result for streaming
                    comm.send({
                        "type": "result",
                        "prompt": message_str,
                        "content": "",
                        "prompt_id": prompt_id,
                        "task_id": task_id,
                        "truncated": tokens_generated >= max_tokens
                    }, dest=0)
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
                    if worker_verbose:
                        logger.info(
                            f"Rank {rank}: Prompt ID {prompt_id}: Full response={response}")
                    content = response["choices"][0].get(
                        "message", {}).get("content", "")
                    if not content and worker_verbose:
                        logger.warning(
                            f"Rank {rank}: Prompt ID {prompt_id}: Empty content in response")
                    tokens_generated += len(content.split())
                    comm.send({
                        "type": "result",
                        "prompt": message_str,
                        "content": content,
                        "prompt_id": prompt_id,
                        "task_id": task_id,
                        "truncated": tokens_generated >= max_tokens
                    }, dest=0)
            except Exception as e:
                error_msg = f"Rank {rank}: Prompt KEID {prompt_id} failed: {str(e)}"
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
