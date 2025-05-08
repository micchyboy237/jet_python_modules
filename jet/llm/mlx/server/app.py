import asyncio
import json
import logging
import time
import uuid
import uvicorn
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import List, Optional, Union, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from mpi4py import MPI
from pydantic import BaseModel
from jet.llm.mlx.server.parallel_stream_script import parallel_stream_generate, parallel_chat_generate
from jet.llm.mlx.mlx_types import ModelType, Message, ModelTypeEnum, RoleMapping, Tool
from jet.llm.mlx.models import AVAILABLE_MODELS, get_model_limits

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

app = FastAPI(title="Parallel MLX Stream Generation Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type", "Accept"],
)

executor = ThreadPoolExecutor()
logger = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    model: ModelType = ModelTypeEnum.LLAMA_3_2_1B_INSTRUCT_4BIT
    prompts: List[str]
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: Optional[float] = None
    repetition_context_size: int = 20
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.0
    logit_bias: Optional[Dict[int, float]] = None
    logprobs: int = -1
    stop: Optional[Union[str, List[str]]] = None
    verbose: bool = False
    worker_verbose: bool = False
    task_id: Optional[str] = None


class ChatRequest(BaseModel):
    model: ModelType = ModelTypeEnum.LLAMA_3_2_1B_INSTRUCT_4BIT
    messages: List[Message]
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: Optional[float] = None
    repetition_context_size: int = 20
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.0
    logit_bias: Optional[Dict[int, float]] = None
    logprobs: int = -1
    stop: Optional[Union[str, List[str]]] = None
    role_mapping: Optional[RoleMapping] = None
    tools: Optional[List[Tool]] = None
    system_prompt: Optional[str] = None
    verbose: bool = False
    worker_verbose: bool = False
    task_id: Optional[str] = None
    session_id: Optional[str] = None


async def stream_generate(request: GenerateRequest, is_chat: bool = False, stream: bool = True):
    task_id = request.task_id or str(uuid.uuid4())
    prompts = request.prompts if not is_chat else [
        json.dumps(request.messages)]
    if not prompts:
        raise HTTPException(
            status_code=400, detail="At least one prompt or message is required")

    # Validate model
    model_key = request.model
    if model_key not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400, detail=f"Invalid model: {model_key}. Must be one of {list(AVAILABLE_MODELS.keys())}")
    model_path = AVAILABLE_MODELS[model_key]
    try:
        max_context, max_embeddings = get_model_limits(model_path)
        if not max_context or not max_embeddings:
            raise ValueError(
                f"Invalid model configuration for {model_key}: max_context={max_context}, max_embeddings={max_embeddings}")
        if request.verbose:
            logger.info(
                f"Model {model_key} validated: path={model_path}, max_context={max_context}, max_embeddings={max_embeddings}")
    except Exception as e:
        logger.error(f"Failed to validate model {model_key}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Model validation failed for {model_key}: {str(e)}")

    num_prompts = len(prompts)
    prompts_per_worker = [[] for _ in range(size - 1)]
    for i, prompt in enumerate(prompts):
        worker_idx = i % (size - 1)
        prompts_per_worker[worker_idx].append(prompt)

    active_workers = set()
    remaining_prompts = {}
    for worker_rank in range(1, size):
        worker_prompts = prompts_per_worker[worker_rank - 1]
        if worker_prompts:
            active_workers.add(worker_rank)
            remaining_prompts[worker_rank] = len(worker_prompts)
            if request.verbose:
                logger.info(
                    f"{time.time()}: Sending {len(worker_prompts)} prompts to worker {worker_rank}")
        comm.send(
            {
                "model": model_path,
                "prompts" if not is_chat else "messages": worker_prompts,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "repetition_penalty": request.repetition_penalty,
                "repetition_context_size": request.repetition_context_size,
                "xtc_probability": request.xtc_probability,
                "xtc_threshold": request.xtc_threshold,
                "logit_bias": request.logit_bias,
                "logprobs": request.logprobs,
                "stop": request.stop,
                "role_mapping": getattr(request, "role_mapping", None),
                "tools": getattr(request, "tools", None),
                "system_prompt": getattr(request, "system_prompt", None),
                "verbose": request.verbose,
                "worker_verbose": request.worker_verbose,
                "task_id": task_id,
                "session_id": getattr(request, "session_id", None),
                "is_chat": is_chat,
                "stream": stream
            },
            dest=worker_rank
        )

    async def stream_results():
        while active_workers:
            for worker_rank in list(active_workers):
                if comm.Iprobe(source=worker_rank, tag=MPI.ANY_TAG):
                    message = comm.recv(source=worker_rank, tag=MPI.ANY_TAG)
                    if request.verbose:
                        logger.info(
                            f"{time.time()}: Received message from worker {worker_rank}: {message}")
                    yield f"{json.dumps(message)}\n"
                    if message["type"] == "result":
                        remaining_prompts[worker_rank] -= 1
                        if request.verbose:
                            logger.info(
                                f"{time.time()}: Worker {worker_rank} has {remaining_prompts[worker_rank]} prompts left")
                        if remaining_prompts[worker_rank] == 0:
                            active_workers.remove(worker_rank)
                            if request.verbose:
                                logger.info(
                                    f"{time.time()}: Worker {worker_rank} has completed all prompts")
            await asyncio.sleep(0.005)

    if stream:
        return StreamingResponse(stream_results(), media_type="application/json")
    else:
        results = []
        while active_workers:
            for worker_rank in list(active_workers):
                if comm.Iprobe(source=worker_rank, tag=MPI.ANY_TAG):
                    message = comm.recv(source=worker_rank, tag=MPI.ANY_TAG)
                    if message["type"] == "result":
                        results.append(message)
                        remaining_prompts[worker_rank] -= 1
                        if remaining_prompts[worker_rank] == 0:
                            active_workers.remove(worker_rank)
            await asyncio.sleep(0.005)
        return results


@app.post("/generate")
async def generate(generate_request: GenerateRequest):
    return await stream_generate(generate_request, is_chat=False, stream=True)


@app.post("/generate_non_stream")
async def generate_non_stream(generate_request: GenerateRequest):
    return await stream_generate(generate_request, is_chat=False, stream=False)


@app.post("/chat")
async def chat(chat_request: ChatRequest):
    return await stream_generate(chat_request, is_chat=True, stream=True)


@app.post("/chat_non_stream")
async def chat_non_stream(chat_request: ChatRequest):
    return await stream_generate(chat_request, is_chat=True, stream=False)


@app.get("/health")
async def health():
    return {"status": "healthy", "rank": rank, "size": size}

if __name__ == "__main__":
    if rank == 0:
        logger.info(f"{time.time()}: Starting Uvicorn server on rank 0")
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=9000,
            reload=False,
            log_level="info",
            proxy_headers=True,
            server_header=False
        )
    else:
        logger.info(f"{time.time()}: Process {rank}: Waiting for MPI tasks")
        while True:
            if comm.Iprobe(source=0):
                task = comm.recv(source=0)
                # Check for either "prompts" or "messages" safely
                prompts = task.get("prompts")
                messages = task.get("messages")
                if prompts or messages:
                    if task["verbose"]:
                        logger.info(
                            f"{time.time()}: Worker {rank} received task with {len(prompts or messages)} items")
                    if task["is_chat"]:
                        parallel_chat_generate(
                            model=task["model"],
                            messages=messages,  # Use messages for chat tasks
                            max_tokens=task["max_tokens"],
                            temperature=task["temperature"],
                            top_p=task["top_p"],
                            repetition_penalty=task["repetition_penalty"],
                            repetition_context_size=task["repetition_context_size"],
                            xtc_probability=task["xtc_probability"],
                            xtc_threshold=task["xtc_threshold"],
                            logit_bias=task["logit_bias"],
                            logprobs=task["logprobs"],
                            stop=task["stop"],
                            role_mapping=task["role_mapping"],
                            tools=task["tools"],
                            system_prompt=task["system_prompt"],
                            verbose=task["verbose"],
                            worker_verbose=task["worker_verbose"],
                            task_id=task["task_id"],
                            session_id=task["session_id"],
                            stream=task["stream"]
                        )
                    else:
                        parallel_stream_generate(
                            model=task["model"],
                            prompts=prompts,  # Use prompts for non-chat tasks
                            max_tokens=task["max_tokens"],
                            temperature=task["temperature"],
                            top_p=task["top_p"],
                            repetition_penalty=task["repetition_penalty"],
                            repetition_context_size=task["repetition_context_size"],
                            xtc_probability=task["xtc_probability"],
                            xtc_threshold=task["xtc_threshold"],
                            logit_bias=task["logit_bias"],
                            logprobs=task["logprobs"],
                            stop=task["stop"],
                            verbose=task["verbose"],
                            worker_verbose=task["worker_verbose"],
                            task_id=task["task_id"],
                            stream=task["stream"]
                        )
            time.sleep(0.1)
