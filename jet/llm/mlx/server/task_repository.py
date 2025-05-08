import psycopg
from typing import Dict, List, Optional
from datetime import datetime
from jet.db.postgres import PostgresDB
from jet.logger import logger


class TaskRepository:
    """A repository class for task-specific database operations."""

    def __init__(self):
        self.db = PostgresDB()
        self.initialize_task_schema()

    def initialize_task_schema(self) -> None:
        """Initialize the database schema for task management."""
        with self.db.connect_default_db() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS tasks (
                            task_id TEXT PRIMARY KEY,
                            model TEXT NOT NULL,
                            is_chat BOOLEAN NOT NULL,
                            stream BOOLEAN NOT NULL,
                            status TEXT NOT NULL,
                            created_at TIMESTAMP NOT NULL,
                            updated_at TIMESTAMP NOT NULL
                        );
                        CREATE TABLE IF NOT EXISTS prompts (
                            prompt_id TEXT PRIMARY KEY,
                            task_id TEXT REFERENCES tasks(task_id),
                            prompt TEXT NOT NULL,
                            status TEXT NOT NULL,
                            error TEXT,
                            response TEXT,
                            tokens_generated INTEGER,
                            created_at TIMESTAMP NOT NULL,
                            updated_at TIMESTAMP NOT NULL
                        );
                    """)
                except psycopg.Error as e:
                    raise Exception(
                        f"Failed to initialize task schema: {str(e)}")

    def save_task(self, task: Dict) -> None:
        """Save a task and its prompts to the database."""
        with self.db.connect_default_db() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("""
                        INSERT INTO tasks (task_id, model, is_chat, stream, status, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        task["task_id"],
                        task["model"],
                        task["is_chat"],
                        task["stream"],
                        task["status"],
                        datetime.fromtimestamp(task["created_at"]),
                        datetime.now()
                    ))
                    for prompt_id, prompt_data in task["prompts"].items():
                        cur.execute("""
                            INSERT INTO prompts (prompt_id, task_id, prompt, status, error, response, tokens_generated, created_at, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            prompt_id,
                            task["task_id"],
                            prompt_data["prompt"],
                            prompt_data["status"],
                            prompt_data.get("error"),
                            prompt_data.get("response"),
                            prompt_data.get("tokens_generated", 0),
                            datetime.fromtimestamp(task["created_at"]),
                            datetime.now()
                        ))
                except psycopg.Error as e:
                    logger.error(
                        f"Failed to save task {task['task_id']}: {str(e)}")
                    raise Exception(
                        f"Failed to save task {task['task_id']}: {str(e)}")

    def update_prompt_status(
        self,
        prompt_id: str,
        status: str,
        error: Optional[str] = None,
        response: Optional[str] = None,
        tokens_generated: int = 0
    ) -> None:
        """Update the status of a prompt."""
        with self.db.connect_default_db() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("""
                        UPDATE prompts
                        SET status = %s, error = %s, response = %s, tokens_generated = %s, updated_at = %s
                        WHERE prompt_id = %s
                    """, (
                        status,
                        error,
                        response,
                        tokens_generated,
                        datetime.now(),
                        prompt_id
                    ))
                except psycopg.Error as e:
                    logger.error(
                        f"Failed to update prompt {prompt_id} status: {str(e)}")
                    raise Exception(
                        f"Failed to update prompt {prompt_id} status: {str(e)}")

    def get_task(self, task_id: str) -> Optional[Dict]:
        """Retrieve a task and its prompts from the database."""
        with self.db.connect_default_db() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("""
                        SELECT * FROM tasks WHERE task_id = %s
                    """, (task_id,))
                    task = cur.fetchone()
                    if not task:
                        return None
                    cur.execute("""
                        SELECT * FROM prompts WHERE task_id = %s
                    """, (task_id,))
                    prompts = cur.fetchall()
                    return {
                        "task_id": task["task_id"],
                        "model": task["model"],
                        "is_chat": task["is_chat"],
                        "stream": task["stream"],
                        "status": task["status"],
                        "created_at": task["created_at"].timestamp(),
                        "updated_at": task["updated_at"].timestamp(),
                        "prompts": {
                            prompt["prompt_id"]: {
                                "prompt": prompt["prompt"],
                                "status": prompt["status"],
                                "error": prompt["error"],
                                "response": prompt["response"],
                                "tokens_generated": prompt["tokens_generated"],
                                "created_at": prompt["created_at"].timestamp(),
                                "updated_at": prompt["updated_at"].timestamp()
                            } for prompt in prompts
                        }
                    }
                except psycopg.Error as e:
                    logger.error(
                        f"Failed to retrieve task {task_id}: {str(e)}")
                    raise Exception(
                        f"Failed to retrieve task {task_id}: {str(e)}")

    def get_all_tasks(self, limit: int = 100) -> List[Dict]:
        """Retrieve all tasks from the database."""
        with self.db.connect_default_db() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("""
                        SELECT * FROM tasks LIMIT %s
                    """, (limit,))
                    tasks = cur.fetchall()
                    result = []
                    for task in tasks:
                        cur.execute("""
                            SELECT * FROM prompts WHERE task_id = %s
                        """, (task["task_id"],))
                        prompts = cur.fetchall()
                        result.append({
                            "task_id": task["task_id"],
                            "model": task["model"],
                            "is_chat": task["is_chat"],
                            "stream": task["stream"],
                            "status": task["status"],
                            "created_at": task["created_at"].timestamp(),
                            "updated_at": task["updated_at"].timestamp(),
                            "prompts": {
                                prompt["prompt_id"]: {
                                    "prompt": prompt["prompt"],
                                    "status": prompt["status"],
                                    "error": prompt["error"],
                                    "response": prompt["response"],
                                    "tokens_generated": prompt["tokens_generated"],
                                    "created_at": prompt["created_at"].timestamp(),
                                    "updated_at": prompt["updated_at"].timestamp()
                                } for prompt in prompts
                            }
                        })
                    return result
                except psycopg.Error as e:
                    logger.error(f"Failed to retrieve all tasks: {str(e)}")
                    raise Exception(f"Failed to retrieve all tasks: {str(e)}")

    def cleanup_old_tasks(self, hours: int = 1) -> None:
        """Remove tasks older than specified hours that are completed."""
        with self.db.connect_default_db() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("""
                        DELETE FROM prompts
                        WHERE task_id IN (
                            SELECT task_id FROM tasks
                            WHERE status = 'completed'
                            AND updated_at < NOW() - INTERVAL '%s hours'
                        )
                    """, (hours,))
                    cur.execute("""
                        DELETE FROM tasks
                        WHERE status = 'completed'
                        AND updated_at < NOW() - INTERVAL '%s hours'
                    """, (hours,))
                except psycopg.Error as e:
                    logger.error(f"Failed to cleanup old tasks: {str(e)}")
                    raise Exception(f"Failed to cleanup old tasks: {str(e)}")
