import asyncio
from typing import List, Optional
import uuid
import json
import time
from concurrent.futures import ThreadPoolExecutor
from mpi4py import MPI
from jet.llm.mlx.mlx_types import ModelType
from jet.llm.mlx.server.parallel_stream_script import parallel_stream_generate
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from jet.logger import logger

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Process {rank}: Starting application")
    yield
    logger.info(f"Process {rank}: Shutting down application")

app.router.lifespan_context = lifespan


class GenerateRequest(BaseModel):
    model: ModelType
    prompts: List[str]
    max_tokens: int = 100
    temp: float = 0.7
    verbose: bool = False
    task_id: Optional[str] = None


# Thread pool for synchronous MPI calls
executor = ThreadPoolExecutor(max_workers=1)


def receive_message(worker_rank):
    """Run synchronous MPI receive in a separate thread."""
    if comm.Iprobe(source=worker_rank):
        message = comm.recv(source=worker_rank)
        return message
    return None


async def run_mpi_task(prompts, model, max_tokens, temp, verbose, task_id):
    async def stream_output():
        if rank == 0:  # Main process handles receiving and streaming
            try:
                # Distribute prompts to workers
                prompts_per_worker = [[] for _ in range(size - 1)]
                for i, prompt in enumerate(prompts):
                    # Distribute to ranks 1 to size-1
                    worker_rank = (i % (size - 1)) + 1
                    prompts_per_worker[worker_rank - 1].append(prompt)

                # Track remaining prompts per worker and initialize active workers
                remaining_prompts = {}
                active_workers = set()
                for i in range(size - 1):
                    if prompts_per_worker[i]:  # Only include workers with prompts
                        worker_rank = i + 1
                        remaining_prompts[worker_rank] = len(
                            prompts_per_worker[i])
                        active_workers.add(worker_rank)
                logger.debug(f"Initial remaining prompts: {remaining_prompts}")
                logger.debug(f"Active workers: {active_workers}")

                # Send prompts to workers
                for worker_rank in range(1, size):
                    worker_prompts = prompts_per_worker[worker_rank - 1]
                    logger.debug(
                        f"Sending {len(worker_prompts)} prompts to worker {worker_rank}")
                    comm.send({
                        "model": model,
                        "prompts": worker_prompts,
                        "max_tokens": max_tokens,
                        "temp": temp,
                        "verbose": verbose,
                        "task_id": task_id or str(uuid.uuid4())
                    }, dest=worker_rank)

                # Receive and yield chunks from workers
                while active_workers:
                    for worker_rank in list(active_workers):
                        # Run synchronous MPI receive in a thread to avoid blocking asyncio
                        message = await asyncio.get_event_loop().run_in_executor(
                            executor, receive_message, worker_rank
                        )
                        if message:
                            logger.debug(
                                f"Received message from worker {worker_rank}: {message}")
                            yield json.dumps(message) + "\n"
                            if message["type"] == "result" or message["type"] == "error":
                                # Decrement the remaining prompt count for this worker
                                remaining_prompts[worker_rank] -= 1
                                logger.debug(
                                    f"Worker {worker_rank} has {remaining_prompts[worker_rank]} prompts left")
                                if remaining_prompts[worker_rank] <= 0:
                                    logger.debug(
                                        f"Worker {worker_rank} has completed all prompts")
                                    active_workers.discard(worker_rank)
                    # Prevent tight loop from consuming CPU
                    await asyncio.sleep(0.01)
            except Exception as e:
                error_message = f"Stream error: {str(e)}"
                logger.error(error_message)
                yield json.dumps({
                    "type": "error",
                    "message": error_message,
                    "prompt_id": None,
                    "task_id": task_id
                }) + "\n"
        else:  # Worker processes
            try:
                # Receive task from main process
                task = comm.recv(source=0)
                logger.debug(
                    f"Worker {rank} received task with {len(task['prompts'])} prompts")
                # Process task
                parallel_stream_generate(
                    model_name=task["model"],
                    prompts=task["prompts"],
                    max_tokens=task["max_tokens"],
                    temp=task["temp"],
                    verbose=task["verbose"],
                    task_id=task["task_id"]
                )
            except Exception as e:
                logger.error(f"Rank {rank}: Worker error: {str(e)}")
                comm.send({
                    "type": "error",
                    "message": str(e),
                    "prompt_id": None,
                    "task_id": task.get("task_id")
                }, dest=0)

    return stream_output


@app.post("/generate")
async def generate(request: GenerateRequest):
    stream_gen = await run_mpi_task(
        model=request.model,
        prompts=request.prompts,
        max_tokens=request.max_tokens,
        temp=request.temp,
        verbose=request.verbose,
        task_id=request.task_id
    )
    return StreamingResponse(
        stream_gen(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
    )

if __name__ == "__main__":
    if rank == 0:
        logger.info("Starting Uvicorn server on rank 0")
        import uvicorn
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
        logger.info(f"Process {rank}: Waiting for MPI tasks")
        # Workers stay in a loop to process tasks
        while True:
            if comm.Iprobe(source=0):
                task = comm.recv(source=0)
                logger.debug(
                    f"Worker {rank} received task in main loop with {len(task['prompts'])} prompts")
                parallel_stream_generate(
                    model_name=task["model"],
                    prompts=task["prompts"],
                    max_tokens=task["max_tokens"],
                    temp=task["temp"],
                    verbose=task["verbose"],
                    task_id=task["task_id"]
                )
            time.sleep(0.1)  # Prevent busy-waiting
