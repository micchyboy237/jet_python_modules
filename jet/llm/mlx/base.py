import uuid
import json
import time
from typing import Dict, List, Optional, Union, Literal, TypedDict, Any, Iterator
from dataclasses import dataclass
import psycopg
from psycopg.rows import dict_row
# Assuming MLXLMClient is in a separate module
from .client import MLXLMClient, CompletionResponse, Message, RoleMapping, Tool

# Typed dictionaries for structured data (reused from MLXLMClient for consistency)


class Message(TypedDict):
    role: str
    content: str


class RoleMapping(TypedDict, total=False):
    system_prompt: str
    system: str
    user: str
    assistant: str
    stop: str


class Tool(TypedDict):
    type: str
    function: Dict[str, Any]


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

    @dataclass
    class DBConfig:
        dbname: str
        user: str
        password: str
        host: str
        port: str
        session_id: Optional[str] = None

    @dataclass
    class Config:
        model: str
        adapter_path: Optional[str] = None
        draft_model: Optional[str] = None
        trust_remote_code: bool = False
        chat_template: Optional[str] = None
        use_default_chat_template: bool = True

    def __init__(
        self,
        model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        adapter_path: Optional[str] = None,
        draft_model: Optional[str] = None,
        trust_remote_code: bool = False,
        chat_template: Optional[str] = None,
        use_default_chat_template: bool = True,
        config: Optional[Config] = None,
        db_config: Optional[DBConfig] = None,
    ):
        """Initialize the MLX client with configuration and optional database."""
        if config is None:
            config = self.Config(
                model=model,
                adapter_path=adapter_path,
                draft_model=draft_model,
                trust_remote_code=trust_remote_code,
                chat_template=chat_template,
                use_default_chat_template=use_default_chat_template,
            )

        # Initialize MLXLMClient
        self.client = MLXLMClient(MLXLMClient.Config(
            model=config.model,
            adapter_path=config.adapter_path,
            draft_model=config.draft_model,
            trust_remote_code=config.trust_remote_code,
            chat_template=config.chat_template,
            use_default_chat_template=config.use_default_chat_template
        ))

        # Initialize chat history
        if db_config:
            self.history = ChatHistory(
                dbname=db_config.dbname,
                user=db_config.user,
                password=db_config.password,
                host=db_config.host,
                port=db_config.port,
                session_id=db_config.session_id
            )
        else:
            self.history = ChatHistory()

    def chat(
        self,
        message: Union[str, List[Message]],
        model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        draft_model: Optional[str] = None,
        adapter: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        repetition_context_size: int = 20,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        logit_bias: Optional[Dict[int, float]] = None,
        logprobs: int = -1,
        stop: Optional[Union[str, List[str]]] = None,
        role_mapping: Optional[RoleMapping] = None,
        tools: Optional[List[Tool]] = None,
        system_prompt: Optional[str] = None
    ) -> Union[CompletionResponse, List[CompletionResponse]]:
        """Generate a chat completion with history management."""
        # Prepare messages with history
        messages: List[Message] = []
        if system_prompt and not any(msg["role"] == "system" for msg in self.history.get_messages()):
            self.history.add_message("system", system_prompt)

        # Handle message input: str or List[Message]
        if isinstance(message, str):
            self.history.add_message("user", message)
        elif isinstance(message, list):
            for msg in message:
                if "role" in msg and "content" in msg:
                    self.history.add_message(msg["role"], msg["content"])
                else:
                    raise ValueError(
                        "Each message in the list must have 'role' and 'content' keys")
        else:
            raise TypeError(
                "message must be a string or a list of Message dictionaries")

        messages = self.history.get_messages()

        # Call MLXLMClient.chat
        response = self.client.chat(
            messages=messages,
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
        if isinstance(response, dict) and response.get("choices"):
            assistant_content = response["choices"][0].get(
                "message", {}).get("content", "")
            if assistant_content:
                self.history.add_message("assistant", assistant_content)

        return response

    def stream_chat(
        self,
        message: Union[str, List[Message]],
        model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        draft_model: Optional[str] = None,
        adapter: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        repetition_context_size: int = 20,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        logit_bias: Optional[Dict[int, float]] = None,
        logprobs: int = -1,
        stop: Optional[Union[str, List[str]]] = None,
        role_mapping: Optional[RoleMapping] = None,
        tools: Optional[List[Tool]] = None,
        system_prompt: Optional[str] = None
    ) -> Iterator[Union[CompletionResponse, List[CompletionResponse]]]:
        """Stream chat completions with history management."""
        # Prepare messages with history
        messages: List[Message] = []
        if system_prompt and not any(msg["role"] == "system" for msg in self.history.get_messages()):
            self.history.add_message("system", system_prompt)

        # Handle message input: str or List[Message]
        if isinstance(message, str):
            self.history.add_message("user", message)
        elif isinstance(message, list):
            for msg in message:
                if "role" in msg and "content" in msg:
                    self.history.add_message(msg["role"], msg["content"])
                else:
                    raise ValueError(
                        "Each message in the list must have 'role' and 'content' keys")
        else:
            raise TypeError(
                "message must be a string or a list of Message dictionaries")

        messages = self.history.get_messages()

        # Stream responses
        assistant_content = ""
        for response in self.client.stream_chat(
            messages=messages,
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
        if assistant_content:
            self.history.add_message("assistant", assistant_content)

    def generate(
        self,
        prompt: str,
        model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        draft_model: Optional[str] = None,
        adapter: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        repetition_context_size: int = 20,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        logit_bias: Optional[Dict[int, float]] = None,
        logprobs: int = -1,
        stop: Optional[Union[str, List[str]]] = None
    ) -> CompletionResponse:
        """Generate a text completion (no history)."""
        return self.client.generate(
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

    def stream_generate(
        self,
        prompt: str,
        model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        draft_model: Optional[str] = None,
        adapter: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        repetition_context_size: int = 20,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        logit_bias: Optional[Dict[int, float]] = None,
        logprobs: int = -1,
        stop: Optional[Union[str, List[str]]] = None
    ) -> Iterator[CompletionResponse]:
        """Stream text completions (no history)."""
        return self.client.stream_generate(
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
        )

    def clear_history(self):
        """Clear the chat history."""
        self.history.clear()
