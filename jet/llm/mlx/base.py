import uuid
import json
import time
from typing import Dict, List, Optional, Union, Literal, TypedDict, Any, Iterator
from dataclasses import dataclass
from jet.llm.mlx.config import DEFAULT_MODEL
from jet.llm.mlx.logger_utils import ChatLogger
from jet.llm.mlx.mlx_types import ModelKey, ModelType
from jet.llm.mlx.models import resolve_model
from jet.llm.mlx.utils import get_model_max_tokens
from jet.llm.mlx.token_utils import count_tokens, get_tokenizer_fn, merge_texts
import psycopg
from psycopg.rows import dict_row
# Assuming MLXLMClient is in a separate module
from .client import MLXLMClient, ModelsResponse, CompletionResponse, Message, RoleMapping, Tool


# Typed dictionaries for structured data (reused from MLXLMClient for consistency)


class PostgresChatMessageHistory:
    """Class to manage chat message history in a PostgreSQL database using psycopg."""

    def __init__(self, dbname: str, user: str, password: str,
                 host: str, port: str, session_id: Optional[str] = None):
        """Initialize with PostgreSQL connection parameters and session ID."""
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.session_id = session_id or str(uuid.uuid4())
        self._initialize_table()

    def _initialize_table(self):
        """Create the messages table if it doesn't exist."""
        with psycopg.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) NOT NULL,
                        role VARCHAR(50) NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()

    def add_message(self, message: Message):
        """Add a message to the database."""
        with psycopg.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO messages (session_id, role, content)
                    VALUES (%s, %s, %s);
                """, (self.session_id, message["role"], message["content"]))
                conn.commit()

    def get_messages(self) -> List[Message]:
        """Retrieve all messages for the session."""
        with psycopg.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            row_factory=dict_row
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT role, content
                    FROM messages
                    WHERE session_id = %s
                    ORDER BY timestamp ASC;
                """, (self.session_id,))
                return [{"role": row["role"], "content": row["content"]} for row in cur.fetchall()]

    def clear(self):
        """Clear all messages for the session."""
        with psycopg.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM messages
                    WHERE session_id = %s;
                """, (self.session_id,))
                conn.commit()


class ChatHistory:
    """Class to manage chat history, with optional PostgreSQL persistence."""

    def __init__(self, dbname: Optional[str] = None, user: Optional[str] = None,
                 password: Optional[str] = None, host: Optional[str] = None,
                 port: Optional[str] = None, session_id: Optional[str] = None):
        """Initialize with optional PostgreSQL connection parameters and session ID."""
        self.session_id = session_id or str(uuid.uuid4())
        self.use_db = all([dbname, user, password, host, port])

        if self.use_db:
            self.db_history = PostgresChatMessageHistory(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port,
                session_id=self.session_id
            )
            self.messages: List[Message] = self.db_history.get_messages()
        else:
            self.db_history = None
            self.messages: List[Message] = []

    def add_message(self, role: str, content: str):
        """Add a message to the history and persist it if using database."""
        message = {"role": role, "content": content}
        self.messages.append(message)
        if self.use_db and self.db_history:
            self.db_history.add_message(message)

    def get_messages(self) -> List[Message]:
        """Return the current list of messages."""
        return self.messages

    def clear(self):
        """Clear the in-memory and persistent history if using database."""
        self.messages = []
        if self.use_db and self.db_history:
            self.db_history.clear()


class MLX:
    """Wrapper class for MLXLMClient with chat history management."""

    def __init__(
        self,
        # Model Config
        model: ModelType = DEFAULT_MODEL,
        adapter_path: Optional[str] = None,
        draft_model: Optional[ModelType] = None,
        trust_remote_code: bool = False,
        chat_template: Optional[str] = None,
        use_default_chat_template: bool = True,
        # DB Config
        dbname: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
        session_id: Optional[str] = None,
        with_history: bool = False,
        seed: Optional[int] = None,
        log_dir: Optional[str] = None,
    ):
        """Initialize the MLX client with configuration and optional database."""
        self.model_path = resolve_model(model)
        self.with_history = with_history  # Store the with_history flag
        self.log_dir = log_dir
        # Initialize MLXLMClient
        self.client = MLXLMClient(
            model=model,
            adapter_path=adapter_path,
            draft_model=draft_model,
            trust_remote_code=trust_remote_code,
            chat_template=chat_template,
            use_default_chat_template=use_default_chat_template,
            seed=seed,
        )
        self.model = self.client.model_provider.model
        self.tokenizer = self.client.model_provider.tokenizer

        # Initialize chat history
        if with_history and dbname:
            self.history = ChatHistory(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port,
                session_id=session_id
            )
        else:
            self.history = ChatHistory()

    def get_models(self) -> ModelsResponse:
        return self.client.get_models()

    def chat(
        self,
        messages: Union[str, List[Message]],
        model: ModelType = DEFAULT_MODEL,
        draft_model: Optional[ModelType] = None,
        adapter: Optional[str] = None,
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
        log_dir: Optional[str] = None
    ) -> Union[CompletionResponse, List[CompletionResponse]]:
        """Generate a chat completion with history management."""
        # Prepare messages with history
        if system_prompt and not any(msg["role"] == "system" for msg in self.history.get_messages()):
            if self.with_history:
                self.history.add_message("system", system_prompt)

        # Handle messages input: str or List[Message]
        if isinstance(messages, str):
            if self.with_history:
                self.history.add_message("user", messages)
        elif isinstance(messages, list):
            for msg in messages:
                if "role" in msg and "content" in msg:
                    if self.with_history:
                        self.history.add_message(msg["role"], msg["content"])
                else:
                    raise ValueError(
                        "Each message in the list must have 'role' and 'content' keys")
        else:
            raise TypeError(
                "messages must be a string or a list of Message dictionaries")

        all_messages = self.history.get_messages() if self.with_history else (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        ) + ([{"role": "user", "content": messages}] if isinstance(messages, str) else messages)

        if max_tokens == -1:
            # Set remaining tokens as max tokens
            max_tokens = self.get_remaining_tokens(all_messages)

        # Call MLXLMClient.chat
        response = self.client.chat(
            messages=all_messages,
            model=model,
            draft_model=draft_model,
            adapter=adapter,
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
            stream=False,
            role_mapping=role_mapping,
            tools=tools
        )

        # Add assistant response to history
        if self.with_history and isinstance(response, dict) and response.get("choices"):
            assistant_content = response["choices"][0].get(
                "message", {}).get("content", "")
            if assistant_content:
                self.history.add_message("assistant", assistant_content)

        # Log interaction
        log_dir = log_dir or self.log_dir
        if log_dir:
            ChatLogger(log_dir, method="chat").log_interaction(
                all_messages, response)

        return response

    def stream_chat(
        self,
        messages: Union[str, List[Message]],
        model: ModelType = DEFAULT_MODEL,
        draft_model: Optional[ModelType] = None,
        adapter: Optional[str] = None,
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
        log_dir: Optional[str] = None
    ) -> Iterator[Union[CompletionResponse, List[CompletionResponse]]]:
        """Stream chat completions with history management."""
        # Prepare messages with history
        if system_prompt and not any(msg["role"] == "system" for msg in self.history.get_messages()):
            if self.with_history:
                self.history.add_message("system", system_prompt)

        # Handle messages input: str or List[Message]
        if isinstance(messages, str):
            if self.with_history:
                self.history.add_message("user", messages)
        elif isinstance(messages, list):
            for msg in messages:
                if "role" in msg and "content" in msg:
                    if self.with_history:
                        self.history.add_message(msg["role"], msg["content"])
                else:
                    raise ValueError(
                        "Each message in the list must have 'role' and 'content' keys")
        else:
            raise TypeError(
                "messages must be a string or a list of Message dictionaries")

        all_messages = self.history.get_messages() if self.with_history else (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        ) + ([{"role": "user", "content": messages}] if isinstance(messages, str) else messages)

        if max_tokens == -1:
            # Set remaining tokens as max tokens
            max_tokens = self.get_remaining_tokens(all_messages)

        # Stream responses
        assistant_content = ""
        for response in self.client.stream_chat(
            messages=all_messages,
            model=model,
            draft_model=draft_model,
            adapter=adapter,
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
            tools=tools
        ):
            if response.get("choices"):
                content = response["choices"][0].get(
                    "message", {}).get("content", "")
                assistant_content += content
            yield response

        # Add assistant response to history
        if self.with_history and assistant_content:
            self.history.add_message("assistant", assistant_content)

        # Log interaction
        log_dir = log_dir or self.log_dir
        if log_dir:
            ChatLogger(log_dir, method="stream_chat").log_interaction(
                all_messages, response)

    def generate(
        self,
        prompt: str,
        model: ModelType = DEFAULT_MODEL,
        draft_model: Optional[ModelType] = None,
        adapter: Optional[str] = None,
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
        log_dir: Optional[str] = None
    ) -> CompletionResponse:
        """Generate a text completion (no history)."""
        response = self.client.generate(
            prompt=prompt,
            model=model,
            draft_model=draft_model,
            adapter=adapter,
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
            stream=False
        )

        # Log interaction
        log_dir = log_dir or self.log_dir
        if log_dir:
            ChatLogger(log_dir, method="generate").log_interaction(
                prompt, response)

        return response

    def stream_generate(
        self,
        prompt: str,
        model: ModelType = DEFAULT_MODEL,
        draft_model: Optional[ModelType] = None,
        adapter: Optional[str] = None,
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
        log_dir: Optional[str] = None
    ) -> Iterator[CompletionResponse]:
        """Stream text completions (no history)."""
        for response in self.client.stream_generate(
            prompt=prompt,
            model=model,
            draft_model=draft_model,
            adapter=adapter,
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
            yield response

        # Log interaction
        log_dir = log_dir or self.log_dir
        if log_dir:
            ChatLogger(log_dir, method="stream_generate").log_interaction(
                prompt, response)

    def clear_history(self):
        """Clear the chat history."""
        self.history.clear()

    def count_tokens(self, messages: str | List[str] | List[Dict], prevent_total: bool = False) -> int | list[int]:
        return count_tokens(self.model_path, messages, prevent_total)

    def filter_docs(self, messages: str | List[str] | List[Message], chunk_size: int, buffer: int = 1024) -> list[str]:
        """Filter documents to fit within model token limits."""
        # Convert messages to a single string
        if isinstance(messages, str):
            context = messages
        elif isinstance(messages, list):
            if all(isinstance(msg, str) for msg in messages):
                context = "\n\n".join(messages)
            elif all(isinstance(msg, dict) and "content" in msg for msg in messages):
                context = "\n\n".join(msg["content"] for msg in messages)
            else:
                raise ValueError(
                    "Messages list must contain strings or Message dictionaries")
        else:
            raise TypeError(
                "Messages must be a string or list of strings/dictionaries")

        # Get model max tokens and reserve buffer
        model_max_tokens = get_model_max_tokens(self.model_path)
        max_tokens = model_max_tokens - buffer

        # Merge texts to fit within token limit
        merged_texts = merge_texts(
            context, self.tokenizer, max_length=chunk_size)

        # Build filtered context
        filtered_contexts = []
        current_token_count = 0

        for text, token_count in zip(merged_texts["texts"], merged_texts["token_counts"]):
            if current_token_count + token_count > max_tokens:
                break
            filtered_contexts.append(text)
            current_token_count += token_count

        return filtered_contexts

    def get_remaining_tokens(self, messages: str | List[str] | List[Message]) -> int:
        model_max_tokens = get_model_max_tokens(self.model_path)
        prompt_tokens = self.count_tokens(messages)
        # Set remaining tokens as max tokens
        remaining_tokens = model_max_tokens - prompt_tokens
        return remaining_tokens
