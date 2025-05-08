import asyncio
import json
import time
import uuid
import uvicorn
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from mpi4py import MPI
from pydantic import BaseModel
from jet.llm.mlx.server.parallel_stream_script import parallel_stream_generate, parallel_chat_generate
from jet.llm.mlx.server.task_manager import TaskManager, TaskStatus
from jet.llm.mlx.mlx_types import ModelType, Message, ModelTypeEnum, RoleMapping, Tool
from jet.llm.mlx.models import AVAILABLE_MODELS, get_model_limits
from jet.logger import logger
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
app = FastAPI(title="Parallel MLX Stream Generation Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "Accept"],
)
executor = ThreadPoolExecutor()
task_manager = TaskManager()


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


async def stream_generate(request: Union[GenerateRequest, ChatRequest], is_chat: bool = False, stream: bool = True):
    task_id = request.task_id or str(uuid.uuid4())
    prompts = []
    if is_chat:
        if not hasattr(request, 'messages') or not request.messages:
            raise HTTPException(
                status_code=400, detail="Messages are required for chat requests")
        prompts = [json.dumps(request.messages)]
    else:
        if not hasattr(request, 'prompts') or not request.prompts:
            raise HTTPException(
                status_code=400, detail="Prompts are required for generate requests")
        prompts = request.prompts
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
    prompt_ids = [str(uuid.uuid4()) for _ in prompts]
    try:
        task_manager.create_task(
            task_id=task_id,
            model=model_path,
            is_chat=is_chat,
            stream=stream,
            prompts=prompts,
            prompt_ids=prompt_ids,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            repetition_context_size=request.repetition_context_size,
            xtc_probability=request.xtc_probability,
            xtc_threshold=request.xtc_threshold,
            logit_bias=request.logit_bias,
            logprobs=request.logprobs,
            stop=request.stop,
            verbose=request.verbose,
            worker_verbose=request.worker_verbose,
            role_mapping=getattr(request, "role_mapping", None),
            tools=getattr(request, "tools", None),
            system_prompt=getattr(request, "system_prompt", None),
            session_id=getattr(request, "session_id", None)
        )
    except Exception as e:
        logger.error(f"Failed to create task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create task: {str(e)}")
    for worker_rank in range(1, size):
        while comm.Iprobe(source=worker_rank, tag=MPI.ANY_TAG):
            comm.recv(source=worker_rank, tag=MPI.ANY_TAG)
            if request.verbose:
                logger.info(f"Cleared stale message from worker {worker_rank}")
    prompts_per_worker = [[] for _ in range(size - 1)]
    for i, (prompt, prompt_id) in enumerate(zip(prompts, prompt_ids)):
        worker_idx = i % (size - 1)
        prompts_per_worker[worker_idx].append(
            {"prompt": prompt, "prompt_id": prompt_id})
    active_workers = set()
    remaining_prompts = {}
    for worker_rank in range(1, size):
        worker_prompts = prompts_per_worker[worker_rank - 1]
        if worker_prompts:
            active_workers.add(worker_rank)
            remaining_prompts[worker_rank] = len(worker_prompts)
            if request.verbose:
                logger.info(
                    f"{time.time()}: Sending {len(worker_prompts)} prompts to worker {worker_rank}: {[p['prompt'][:50] for p in worker_prompts]}")
            task = {
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
            }
            comm.send(task, dest=worker_rank)

    async def stream_results():
        while active_workers:
            for worker_rank in list(active_workers):
                if comm.Iprobe(source=worker_rank, tag=MPI.ANY_TAG):
                    message = comm.recv(source=worker_rank, tag=MPI.ANY_TAG)
                    if message.get("task_id") != task_id:
                        if request.verbose:
                            logger.warning(
                                f"Ignoring stale message from worker {worker_rank} with task_id {message.get('task_id')}")
                        continue
                    if request.verbose:
                        logger.info(
                            f"{time.time()}: Received message from worker {worker_rank}: {message}")
                    prompt_id = message.get("prompt_id")
                    if not task_manager.validate_prompt_id(task_id, prompt_id):
                        logger.error(
                            f"Invalid prompt_id {prompt_id} for task {task_id}")
                        continue
                    try:
                        task_manager.process_message(
                            task_id, prompt_id, message)
                    except Exception as e:
                        logger.error(
                            f"Failed to process message for prompt {prompt_id}: {str(e)}")
                        continue
                    if message["type"] == "result":
                        remaining_prompts[worker_rank] -= 1
                        if remaining_prompts[worker_rank] == 0:
                            active_workers.remove(worker_rank)
                            if request.verbose:
                                logger.info(
                                    f"{time.time()}: Worker {worker_rank} has completed all prompts")
                    yield f"{json.dumps(message)}\n"
            await asyncio.sleep(0.005)
        try:
            task_manager.complete_task(task_id)
        except Exception as e:
            logger.error(f"Failed to complete task {task_id}: {str(e)}")
    if stream:
        return StreamingResponse(stream_results(), media_type="application/json")
    else:
        results = []
        while active_workers:
            for worker_rank in list(active_workers):
                if comm.Iprobe(source=worker_rank, tag=MPI.ANY_TAG):
                    message = comm.recv(source=worker_rank, tag=MPI.ANY_TAG)
                    if message.get("task_id") != task_id:
                        if request.verbose:
                            logger.warning(
                                f"Ignoring stale message from worker {worker_rank} with task_id {message.get('task_id')}")
                        continue
                    prompt_id = message.get("prompt_id")
                    if not task_manager.validate_prompt_id(task_id, prompt_id):
                        logger.error(
                            f"Invalid prompt_id {prompt_id} for task {task_id}")
                        continue
                    try:
                        task_manager.process_message(
                            task_id, prompt_id, message)
                    except Exception as e:
                        logger.error(
                            f"Failed to process message for prompt {prompt_id}: {str(e)}")
                        continue
                    if message["type"] == "result":
                        results.append(message)
                        remaining_prompts[worker_rank] -= 1
                        if remaining_prompts[worker_rank] == 0:
                            active_workers.remove(worker_rank)
                    elif message["type"] == "error":
                        pass
            await asyncio.sleep(0.005)
        try:
            task_manager.complete_task(task_id)
        except Exception as e:
            logger.error(f"Failed to complete task {task_id}: {str(e)}")
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


@app.get("/tasks")
async def get_tasks():
    try:
        return {"tasks": task_manager.get_all_tasks()}
    except Exception as e:
        logger.error(f"Failed to retrieve tasks: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve tasks: {str(e)}")


@app.get("/task/{task_id}")
async def get_task(task_id: str):
    try:
        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve task: {str(e)}")


@app.post("/rerun_failed/{task_id}")
async def rerun_failed(task_id: str):
    """Rerun failed prompts for a specific task."""
    try:
        task_manager.rerun_task(task_id, only_failed=True)
        return {"message": f"Successfully triggered rerun of failed prompts for task {task_id}"}
    except ValueError as e:
        logger.error(f"Invalid task {task_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            f"Failed to rerun failed prompts for task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to rerun failed prompts: {str(e)}")


@app.post("/rerun_pending/{task_id}")
async def rerun_pending(task_id: str):
    """Rerun a specific task with pending, processing, or failed prompts."""
    try:
        task_manager.rerun_task(task_id, only_failed=False)
        return {"message": f"Successfully triggered rerun for task {task_id}"}
    except ValueError as e:
        logger.error(f"Invalid task {task_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to rerun task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to rerun task {task_id}: {str(e)}")

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
                prompts = task.get("prompts")
                messages = task.get("messages")
                if prompts or messages:
                    if task["verbose"]:
                        logger.info(
                            f"{time.time()}: Worker {rank} received task {task['task_id']} with {len(prompts or messages)} items")
                    if task["is_chat"]:
                        parallel_chat_generate(
                            model=task["model"],
                            messages=messages,
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
                            prompts=prompts,
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
