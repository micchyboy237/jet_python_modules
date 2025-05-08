from jet.transformers.formatters import format_json
import psycopg
import json
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
        """Initialize the database schema for task management and migrate if needed."""
        with self.db.connect_default_db() as conn:
            with conn.cursor() as cur:
                try:
                    # Create tasks table if it doesn't exist
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS tasks (
                            task_id TEXT PRIMARY KEY,
                            model TEXT NOT NULL,
                            is_chat BOOLEAN NOT NULL,
                            stream BOOLEAN NOT NULL,
                            status TEXT NOT NULL,
                            created_at TIMESTAMP NOT NULL,
                            updated_at TIMESTAMP NOT NULL
                        )
                    """)
                    # Create prompts table if it doesn't exist
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS prompts (
                            prompt_id TEXT PRIMARY KEY,
                            task_id TEXT REFERENCES tasks(task_id) ON DELETE CASCADE,
                            prompt TEXT NOT NULL,
                            status TEXT NOT NULL,
                            error TEXT,
                            response TEXT,
                            tokens_generated INTEGER DEFAULT 0,
                            created_at TIMESTAMP NOT NULL,
                            updated_at TIMESTAMP NOT NULL
                        )
                    """)
                    # Check existing columns
                    cur.execute("""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = 'tasks'
                    """)
                    existing_columns = {row["column_name"]
                                        for row in cur.fetchall()}

                    # Define new columns with their types and constraints
                    new_columns = [
                        ("max_tokens", "INTEGER NOT NULL DEFAULT 512"),
                        ("temperature", "FLOAT NOT NULL DEFAULT 0.0"),
                        ("top_p", "FLOAT NOT NULL DEFAULT 1.0"),
                        ("repetition_penalty", "FLOAT"),
                        ("repetition_context_size", "INTEGER NOT NULL DEFAULT 20"),
                        ("xtc_probability", "FLOAT NOT NULL DEFAULT 0.0"),
                        ("xtc_threshold", "FLOAT NOT NULL DEFAULT 0.0"),
                        ("logit_bias", "JSONB"),
                        ("logprobs", "INTEGER NOT NULL DEFAULT -1"),
                        ("stop", "JSONB"),
                        ("verbose", "BOOLEAN NOT NULL DEFAULT FALSE"),
                        ("worker_verbose", "BOOLEAN NOT NULL DEFAULT FALSE"),
                        ("role_mapping", "JSONB"),
                        ("tools", "JSONB"),
                        ("system_prompt", "TEXT"),
                        ("session_id", "TEXT")
                    ]

                    # Add new columns only if they don't exist
                    for column_name, column_type in new_columns:
                        if column_name not in existing_columns:
                            logger.info(
                                f"Adding missing column: {column_name} ({column_type})")
                            cur.execute(f"""
                                ALTER TABLE tasks
                                ADD COLUMN {column_name} {column_type}
                            """)
                        else:
                            logger.info(
                                f"Column {column_name} already exists in tasks table, skipping addition")

                    conn.commit()  # Save changes

                except psycopg.Error as e:
                    logger.error(f"Failed to initialize task schema: {str(e)}")
                    raise Exception(
                        f"Failed to initialize task schema: {str(e)}")

    def save_task(self, task: Dict) -> None:
        """Save or update a task and its prompts to the database."""
        with self.db.connect_default_db() as conn:
            with conn.cursor() as cur:
                try:
                    # Upsert task
                    cur.execute("""
                        INSERT INTO tasks (
                            task_id, model, is_chat, stream, status, max_tokens, temperature, top_p,
                            repetition_penalty, repetition_context_size, xtc_probability, xtc_threshold,
                            logit_bias, logprobs, stop, "verbose", worker_verbose, role_mapping, tools,
                            system_prompt, session_id, created_at, updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (task_id)
                        DO UPDATE SET
                            model = EXCLUDED.model,
                            is_chat = EXCLUDED.is_chat,
                            stream = EXCLUDED.stream,
                            status = EXCLUDED.status,
                            max_tokens = EXCLUDED.max_tokens,
                            temperature = EXCLUDED.temperature,
                            top_p = EXCLUDED.top_p,
                            repetition_penalty = EXCLUDED.repetition_penalty,
                            repetition_context_size = EXCLUDED.repetition_context_size,
                            xtc_probability = EXCLUDED.xtc_probability,
                            xtc_threshold = EXCLUDED.xtc_threshold,
                            logit_bias = EXCLUDED.logit_bias,
                            logprobs = EXCLUDED.logprobs,
                            stop = EXCLUDED.stop,
                            "verbose" = EXCLUDED."verbose",
                            worker_verbose = EXCLUDED.worker_verbose,
                            role_mapping = EXCLUDED.role_mapping,
                            tools = EXCLUDED.tools,
                            system_prompt = EXCLUDED.system_prompt,
                            session_id = EXCLUDED.session_id,
                            updated_at = EXCLUDED.updated_at
                    """, (
                        task["task_id"],
                        task["model"],
                        task["is_chat"],
                        task["stream"],
                        task["status"],
                        task.get("max_tokens", 512),
                        task.get("temperature", 0.0),
                        task.get("top_p", 1.0),
                        task.get("repetition_penalty"),
                        task.get("repetition_context_size", 20),
                        task.get("xtc_probability", 0.0),
                        task.get("xtc_threshold", 0.0),
                        json.dumps(task.get("logit_bias")) if task.get(
                            "logit_bias") else None,
                        task.get("logprobs", -1),
                        json.dumps(task.get("stop")) if task.get(
                            "stop") else None,
                        task.get("verbose", False),
                        task.get("worker_verbose", False),
                        json.dumps(task.get("role_mapping")) if task.get(
                            "role_mapping") else None,
                        json.dumps(task.get("tools")) if task.get(
                            "tools") else None,
                        task.get("system_prompt"),
                        task.get("session_id"),
                        datetime.fromtimestamp(task["created_at"]),
                        datetime.now()
                    ))
                    # Upsert prompts
                    for prompt_id, prompt_data in task["prompts"].items():
                        cur.execute("""
                            INSERT INTO prompts (
                                prompt_id, task_id, prompt, status, error,
                                response, tokens_generated, created_at, updated_at
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (prompt_id)
                            DO UPDATE SET
                                prompt = EXCLUDED.prompt,
                                status = EXCLUDED.status,
                                error = EXCLUDED.error,
                                response = EXCLUDED.response,
                                tokens_generated = EXCLUDED.tokens_generated,
                                updated_at = EXCLUDED.updated_at
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
                        SET status = %s,
                            error = %s,
                            response = %s,
                            tokens_generated = %s,
                            updated_at = %s
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
                    task_data = {
                        "task_id": task["task_id"],
                        "model": task["model"],
                        "is_chat": task["is_chat"],
                        "stream": task["stream"],
                        "status": task["status"],
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
                        "verbose": task.get("verbose", False),
                        "worker_verbose": task.get("worker_verbose", False),
                        "role_mapping": task.get("role_mapping"),
                        "tools": task.get("tools"),
                        "system_prompt": task.get("system_prompt"),
                        "session_id": task.get("session_id"),
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
                    return task_data
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
                        SELECT * FROM tasks
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (limit,))
                    tasks = cur.fetchall()
                    result = []
                    for task in tasks:
                        logger.debug(
                            f"DB Task ({task["task_id"]}):\n{task}")
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
                            "verbose": task.get("verbose", False),
                            "worker_verbose": task.get("worker_verbose", False),
                            "role_mapping": task.get("role_mapping"),
                            "tools": task.get("tools"),
                            "system_prompt": task.get("system_prompt"),
                            "session_id": task.get("session_id"),
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
                        DELETE FROM tasks
                        WHERE status = 'completed'
                        AND updated_at < NOW() - INTERVAL '%s hours'
                    """, (hours,))
                except psycopg.Error as e:
                    logger.error(f"Failed to cleanup old tasks: {str(e)}")
                    raise Exception(f"Failed to cleanup old tasks: {str(e)}")
