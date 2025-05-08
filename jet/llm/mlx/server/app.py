import asyncio
import json
import logging
import time
import uuid
import uvicorn
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from mpi4py import MPI
from pydantic import BaseModel
from jet.llm.mlx.server.parallel_stream_script import parallel_stream_generate

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    LLAMA_3_2_1B_INSTRUCT_4BIT = "llama-3.2-1b-instruct-4bit"


class GenerateRequest(BaseModel):
    model: ModelType
    prompts: List[str]
    max_tokens: int = 100
    temp: float = 0.7
    verbose: bool = False
    worker_verbose: bool = False
    task_id: Optional[str] = None


async def stream_generate(generate_request: GenerateRequest):
    task_id = generate_request.task_id or str(uuid.uuid4())
    prompts = generate_request.prompts
    if not prompts:
        raise HTTPException(
            status_code=400, detail="At least one prompt is required")

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
            if generate_request.verbose:
                logger.info(
                    f"{time.time()}: Sending {len(worker_prompts)} prompts to worker {worker_rank}")
        comm.send(
            {
                "model": generate_request.model,
                "prompts": worker_prompts,
                "max_tokens": generate_request.max_tokens,
                "temp": generate_request.temp,
                "verbose": generate_request.verbose,
                "worker_verbose": generate_request.worker_verbose,
                "task_id": task_id
            },
            dest=worker_rank
        )

    async def stream_results():
        while active_workers:
            for worker_rank in list(active_workers):
                if comm.Iprobe(source=worker_rank, tag=MPI.ANY_TAG):
                    message = comm.recv(source=worker_rank, tag=MPI.ANY_TAG)
                    if generate_request.verbose:
                        logger.info(
                            f"{time.time()}: Received message from worker {worker_rank}: {message}")
                    yield f"{json.dumps(message)}\n"
                    if message["type"] == "result":
                        remaining_prompts[worker_rank] -= 1
                        if generate_request.verbose:
                            logger.info(
                                f"{time.time()}: Worker {worker_rank} has {remaining_prompts[worker_rank]} prompts left")
                        if remaining_prompts[worker_rank] == 0:
                            active_workers.remove(worker_rank)
                            if generate_request.verbose:
                                logger.info(
                                    f"{time.time()}: Worker {worker_rank} has completed all prompts")
            await asyncio.sleep(0.005)

    return StreamingResponse(stream_results(), media_type="application/json")


@app.post("/generate")
async def generate(generate_request: GenerateRequest):
    return await stream_generate(generate_request)

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
                if task["prompts"]:
                    if task["verbose"]:
                        logger.info(
                            f"{time.time()}: Worker {rank} received task in main loop with {len(task['prompts'])} prompts")
                    parallel_stream_generate(
                        model_name=task["model"],
                        prompts=task["prompts"],
                        max_tokens=task["max_tokens"],
                        temp=task["temp"],
                        verbose=task["verbose"],
                        worker_verbose=task["worker_verbose"],
                        task_id=task["task_id"]
                    )
            time.sleep(0.1)
