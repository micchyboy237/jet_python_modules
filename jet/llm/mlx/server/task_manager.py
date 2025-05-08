import time
from threading import Lock
from enum import Enum
from typing import Dict, List, Optional, Union
from jet.llm.mlx.server.task_repository import TaskRepository
from jet.logger import logger
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskManager:
    def __init__(self):
        self.repository = TaskRepository()
        self.tasks = {}
        self.lock = Lock()

    def create_task(
        self,
        task_id: str,
        model: str,
        is_chat: bool,
        stream: bool,
        prompts: List[str],
        prompt_ids: List[str],
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
        role_mapping: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> None:
        """Create a new task and save it to the database."""
        with self.lock:
            self.tasks[task_id] = {
                "task_id": task_id,
                "model": model,
                "is_chat": is_chat,
                "stream": stream,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "repetition_context_size": repetition_context_size,
                "xtc_probability": xtc_probability,
                "xtc_threshold": xtc_threshold,
                "logit_bias": logit_bias,
                "logprobs": logprobs,
                "stop": stop,
                "verbose": verbose,
                "worker_verbose": worker_verbose,
                "role_mapping": role_mapping,
                "tools": tools,
                "system_prompt": system_prompt,
                "session_id": session_id,
                "prompts": {
                    prompt_id: {
                        "prompt": prompt,
                        "status": TaskStatus.PENDING,
                        "error": None,
                        "response": "",
                        "tokens_generated": 0
                    }
                    for prompt_id, prompt in zip(prompt_ids, prompts)
                },
                "created_at": time.time(),
                "status": "running"
            }
            try:
                self.repository.save_task(self.tasks[task_id])
            except Exception as e:
                logger.error(
                    f"Failed to save task {task_id} to database: {str(e)}")
                raise

    def rerun_task(self, task_id: str, only_failed: bool = False) -> None:
        """Rerun prompts for a specific task and requeue to MPI."""
        with self.lock:
            if task_id not in self.tasks:
                raise ValueError(f"Task {task_id} not found")
            task = self.tasks[task_id]
            task["status"] = "running"
            prompts_to_rerun = []
            prompt_ids_to_rerun = []
            for prompt_id, prompt_data in task["prompts"].items():
                if only_failed:
                    if prompt_data["status"] == TaskStatus.FAILED:
                        prompt_data["status"] = TaskStatus.PENDING
                        prompt_data["error"] = None
                        prompt_data["response"] = ""
                        prompt_data["tokens_generated"] = 0
                        prompts_to_rerun.append(prompt_data["prompt"])
                        prompt_ids_to_rerun.append(prompt_id)
                else:
                    if prompt_data["status"] in [TaskStatus.PENDING, TaskStatus.PROCESSING, TaskStatus.FAILED]:
                        prompt_data["status"] = TaskStatus.PENDING
                        prompt_data["error"] = None
                        prompt_data["response"] = ""
                        prompt_data["tokens_generated"] = 0
                        prompts_to_rerun.append(prompt_data["prompt"])
                        prompt_ids_to_rerun.append(prompt_id)
            if not prompts_to_rerun:
                logger.info(f"No prompts to rerun for task {task_id}")
                task["status"] = "completed"
                try:
                    self.repository.save_task(task)
                except Exception as e:
                    logger.error(
                        f"Failed to update task {task_id} in database: {str(e)}")
                    raise
                return

            # Requeue to MPI
            prompts_per_worker = [[] for _ in range(size - 1)]
            for i, (prompt, prompt_id) in enumerate(zip(prompts_to_rerun, prompt_ids_to_rerun)):
                worker_idx = i % (size - 1)
                prompts_per_worker[worker_idx].append(
                    {"prompt": prompt, "prompt_id": prompt_id})

            for worker_rank in range(1, size):
                worker_prompts = prompts_per_worker[worker_rank - 1]
                if worker_prompts:
                    task_message = {
                        "model": task["model"],
                        "prompts" if not task["is_chat"] else "messages": worker_prompts,
                        "max_tokens": task.get("max_tokens", 512),
                        "temperature": task.get("temperature", 0.0),
                        "top_p": task.get("top_p", 1.0),
                        "repetition_penalty": task.get("repetition_penalty"),
                        "repetition_context_size": task.get("repetition_context_size", 20),
                        "xtc_probability": task.get("xtc_probability", 0.0),
                        "xtc_threshold": task.get("xtc_threshold", 0.0),
                        "logit_bias": task.get("logit_bias"),
                        "logprobs": task.get("logprobs", -1),
                        "stop": task.get("stop"),
                        "role_mapping": task.get("role_mapping"),
                        "tools": task.get("tools"),
                        "system_prompt": task.get("system_prompt"),
                        "verbose": task.get("verbose", False),
                        "worker_verbose": task.get("worker_verbose", False),
                        "task_id": task_id,
                        "session_id": task.get("session_id"),
                        "is_chat": task["is_chat"],
                        "stream": task["stream"]
                    }
                    if task.get("verbose", False):
                        logger.info(
                            f"Sending {len(worker_prompts)} prompts to worker {worker_rank}")
                    comm.send(task_message, dest=worker_rank)

            try:
                self.repository.save_task(task)
            except Exception as e:
                logger.error(
                    f"Failed to update task {task_id} in database: {str(e)}")
                raise

    def validate_prompt_id(self, task_id: str, prompt_id: str) -> bool:
        """Validate if a prompt_id exists for a given task."""
        with self.lock:
            return task_id in self.tasks and prompt_id in self.tasks[task_id]["prompts"]

    def process_message(self, task_id: str, prompt_id: str, message: Dict) -> None:
        """Process a message from a worker and update task status."""
        with self.lock:
            if task_id not in self.tasks or prompt_id not in self.tasks[task_id]["prompts"]:
                raise ValueError(
                    f"Invalid task_id {task_id} or prompt_id {prompt_id}")
            prompt_data = self.tasks[task_id]["prompts"][prompt_id]
            content = message.get("content", "") if message.get(
                "type") in ["chunk", "result"] else ""
            if not isinstance(content, str):
                logger.warning(
                    f"Invalid content type for prompt {prompt_id}: {type(content)}. Converting to string.")
                content = str(content)
            tokens_generated = prompt_data.get("tokens_generated", 0)
            current_response = prompt_data.get("response") or ""
            if message["type"] == "chunk":
                prompt_data["status"] = TaskStatus.PROCESSING
                prompt_data["response"] = current_response + content
                prompt_data["tokens_generated"] = tokens_generated + \
                    len(content.split())
                try:
                    self.repository.update_prompt_status(
                        prompt_id=prompt_id,
                        status=TaskStatus.PROCESSING,
                        response=prompt_data["response"],
                        tokens_generated=prompt_data["tokens_generated"]
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to update prompt {prompt_id} status: {str(e)}")
                    raise
            elif message["type"] == "result":
                prompt_data["status"] = TaskStatus.COMPLETED
                prompt_data["response"] = current_response + content
                prompt_data["tokens_generated"] = tokens_generated + \
                    len(content.split())
                try:
                    self.repository.update_prompt_status(
                        prompt_id=prompt_id,
                        status=TaskStatus.COMPLETED,
                        response=prompt_data["response"],
                        tokens_generated=prompt_data["tokens_generated"]
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to update prompt {prompt_id} status: {str(e)}")
                    raise
            elif message["type"] == "error":
                prompt_data["status"] = TaskStatus.FAILED
                prompt_data["error"] = message.get("message", "Unknown error")
                try:
                    self.repository.update_prompt_status(
                        prompt_id=prompt_id,
                        status=TaskStatus.FAILED,
                        error=prompt_data["error"]
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to update prompt {prompt_id} status: {str(e)}")
                    raise

    def complete_task(self, task_id: str) -> None:
        """Mark a task as completed and update the database."""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id]["status"] = "completed"
                try:
                    self.repository.save_task(self.tasks[task_id])
                except Exception as e:
                    logger.error(
                        f"Failed to save completed task {task_id}: {str(e)}")
                    raise

    def get_task(self, task_id: str) -> Optional[Dict]:
        """Retrieve a task from the database and update in-memory cache."""
        try:
            task = self.repository.get_task(task_id)
            if task:
                with self.lock:
                    self.tasks[task_id] = task
            return task
        except Exception as e:
            logger.error(
                f"Failed to retrieve task {task_id} from database: {str(e)}")
            raise

    def get_all_tasks(self) -> Dict:
        """Retrieve all tasks from the database and sync in-memory cache."""
        with self.lock:
            try:
                db_tasks = self.repository.get_all_tasks()
                self.tasks.clear()
                for db_task in db_tasks:
                    self.tasks[db_task["task_id"]] = db_task
                self.repository.cleanup_old_tasks()
                return self.tasks
            except Exception as e:
                logger.error(
                    f"Failed to retrieve all tasks from database: {str(e)}")
                raise
