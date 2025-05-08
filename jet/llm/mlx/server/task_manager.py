import time
from threading import Lock
from enum import Enum
from typing import Dict, List, Optional

from jet.llm.mlx.server.task_repository import TaskRepository
from jet.logger import logger


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

    def create_task(self, task_id: str, model: str, is_chat: bool, stream: bool, prompts: List[str], prompt_ids: List[str]) -> None:
        """Create a new task and save it to the database."""
        with self.lock:
            self.tasks[task_id] = {
                "task_id": task_id,
                "model": model,
                "is_chat": is_chat,
                "stream": stream,
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
                for db_task in db_tasks:
                    self.tasks[db_task["task_id"]] = db_task
                self.repository.cleanup_old_tasks()
                return self.tasks
            except Exception as e:
                logger.error(
                    f"Failed to retrieve all tasks from database: {str(e)}")
                raise
