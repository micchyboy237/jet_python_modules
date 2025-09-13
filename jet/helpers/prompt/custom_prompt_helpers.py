from copy import deepcopy
import logging
from typing import Callable, List, Optional
from jet.llm.ollama.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS, DEFAULT_OLLAMA_MODEL
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.helpers.token import OllamaTokenCounter
from llama_index.core.node_parser.text.token import TokenTextSplitter
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.prompts import BasePromptTemplate, ChatPromptTemplate
from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.prompts.prompt_utils import get_empty_prompt_txt
from llama_index.core.prompts.utils import format_string
from llama_index.core.tools.types import BaseTool
from llama_index.core.node_parser.text.utils import truncate_text
from pydantic.fields import Field
from pydantic.main import BaseModel
from jet.llm.ollama.base import Ollama


class OllamaPromptHelperModel(BaseModel):
    """
    Custom class that extends PromptHelper for Ollama LLM models.
    """

    llm: Optional[Ollama] = Field(
        None,
        description="The specific model to use, defined by OLLAMA_MODEL_NAMES.",
    )
    context_window: int = Field(
        default=0,
        description="The maximum context size that will get sent to the LLM.",
    )
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt for LLM calls."
    )


class OllamaPromptHelper(PromptHelper, OllamaPromptHelperModel):
    """
    Custom class that extends PromptHelper for Ollama LLM models.
    """

    def __init__(
        self,
        llm: Optional[OLLAMA_MODEL_NAMES | LLM] = DEFAULT_OLLAMA_MODEL,
        context_window: Optional[int] = None,
        num_output: int = DEFAULT_NUM_OUTPUTS,
        chunk_overlap_ratio: float = 0.1,
        chunk_size_limit: Optional[int] = None,
        tokenizer: Optional[Callable[[str], List]] = None,
        separator: str = " "
    ):
        from jet._token.token_utils import get_model_max_tokens, get_ollama_tokenizer

        # Ensure Pydantic fields and base classes are initialized

        if isinstance(llm, str) and llm in OLLAMA_MODEL_NAMES.__args__:
            ollama_model = Ollama(model=llm)
        elif isinstance(llm, Ollama):
            ollama_model = llm
        else:
            ollama_model = Ollama(model=DEFAULT_OLLAMA_MODEL)

        if not isinstance(context_window, int):
            context_window = get_model_max_tokens(ollama_model.model)

        tokenizer = tokenizer if tokenizer else get_ollama_tokenizer(
            ollama_model.model).encode

        super(PromptHelper, self).__init__(
            context_window=context_window,
            num_output=num_output,
            chunk_overlap_ratio=chunk_overlap_ratio,
            chunk_size_limit=chunk_size_limit,
            tokenizer=tokenizer,
            separator=separator
        )
        super(OllamaPromptHelperModel, self).__init__(
            chunk_size_limit=chunk_size_limit,
            llm=ollama_model,
            context_window=context_window or 0,
            system_prompt=ollama_model.system_prompt,
        )

        self._token_counter = OllamaTokenCounter(tokenizer=tokenizer)

    def _get_available_context_size(self, num_prompt_tokens: int) -> int:
        """Custom context size calculation."""
        context_size_tokens = self.context_window - num_prompt_tokens - self.num_output
        if context_size_tokens < 0:
            raise ValueError("Calculated available context size was negative.")
        return context_size_tokens

    def _get_available_chunk_size(
        self,
        prompt: BasePromptTemplate,
        num_chunks: int = 1,
        padding: int = 5,
        llm: Optional[LLM] = None,
        tools: Optional[List["BaseTool"]] = None,
    ) -> int:
        """Get available chunk size.

        This is calculated as:
            available chunk size = available context window  // number_chunks
                - padding

        Notes:
        - By default, we use padding of 5 (to save space for formatting needs).
        - Available chunk size is further clamped to chunk_size_limit if specified.
        """
        del llm
        tools = self._get_tools_from_llm(llm=self.llm, tools=tools)

        if isinstance(prompt, ChatPromptTemplate):
            messages: List[ChatMessage] = prompt.message_templates

            # account for partial formatting
            partial_messages = []
            for message in messages:
                partial_message = deepcopy(message)

                prompt_kwargs = prompt.kwargs or {}
                partial_message.content = format_string(
                    partial_message.content or "", **prompt_kwargs
                )

                # add to list of partial messages
                partial_messages.append(partial_message)

            num_prompt_tokens = self._token_counter.estimate_tokens_in_messages(
                partial_messages
            )
        else:
            prompt_str = get_empty_prompt_txt(prompt)
            num_prompt_tokens = self._token_counter.get_string_tokens(
                prompt_str)

        num_prompt_tokens += self._token_counter.estimate_tokens_in_tools(
            [x.metadata.to_openai_tool() for x in tools]
        )

        # structured llms cannot have system prompts currently -- check the underlying llm
        num_prompt_tokens += self._token_counter.get_string_tokens(
            self.system_prompt or ""
        )

        available_context_size = self._get_available_context_size(
            num_prompt_tokens)
        result = available_context_size // num_chunks - padding
        if self.chunk_size_limit is not None:
            result = min(result, self.chunk_size_limit)
        return result

    def get_text_splitter_given_prompt(
        self,
        prompt: BasePromptTemplate,
        num_chunks: int = 1,
        padding: int = 5,
        llm: Optional[LLM] = None,
        tools: Optional[List["BaseTool"]] = None,
    ) -> TokenTextSplitter:
        """Get text splitter configured to maximally pack available context window,
        taking into account of given prompt, and desired number of chunks.
        """
        del llm
        chunk_size = self._get_available_chunk_size(
            prompt, num_chunks, padding=padding, llm=self.llm, tools=tools
        )
        if chunk_size <= 0:
            raise ValueError(f"Chunk size {chunk_size} is not positive.")
        chunk_overlap = int(self.chunk_overlap_ratio * chunk_size)
        return TokenTextSplitter(
            separator=self.separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=self._token_counter.tokenizer,
        )

    def truncate(
        self,
        prompt: BasePromptTemplate,
        text_chunks: List[str],
        padding: int = 5,
        llm: Optional[LLM] = None,
        tools: Optional[List["BaseTool"]] = None,
    ) -> List[str]:
        """Truncate text chunks to fit available context window."""
        del llm
        text_splitter = self.get_text_splitter_given_prompt(
            prompt,
            num_chunks=len(text_chunks),
            padding=padding,
            llm=self.llm,
            tools=tools,
        )
        return [truncate_text(chunk, text_splitter) for chunk in text_chunks]

    def repack(
        self,
        prompt: BasePromptTemplate,
        text_chunks: List[str],
        padding: int = 5,
        llm: Optional[LLM] = None,
        tools: Optional[List["BaseTool"]] = None,
    ) -> List[str]:
        """Truncate text chunks to fit available context window."""
        del llm
        text_splitter = self.get_text_splitter_given_prompt(
            prompt,
            num_chunks=len(text_chunks),
            padding=padding,
            llm=self.llm,
            tools=tools,
        )
        return [truncate_text(chunk, text_splitter) for chunk in text_chunks]

    def _get_tools_from_llm(self, llm: Optional[LLM] = None, tools: Optional[List["BaseTool"]] = None) -> List["BaseTool"]:
        """Custom method to handle tools."""
        del llm
        return tools or []
        # from llama_index.core.program.function_program import get_function_tool

        # tools = tools or []
        # if isinstance(llm, StructuredLLM):
        #     tools.append(get_function_tool(llm.output_cls))

        # return tools


# Example Usage
def main():
    # Create a sample prompt template (ChatPromptTemplate)
    prompt = ChatPromptTemplate(
        message_templates=[ChatMessage(
            content="What is the weather like today?")],
    )

    # Initialize OllamaPromptHelper
    helper = OllamaPromptHelper()

    # Sample text chunks to work with
    text_chunks = ["The weather is sunny and warm.",
                   "A great day to go outside!", "Temperature around 75°F."]

    # Truncate text chunks to fit within context size
    truncated_text = helper.truncate(prompt, text_chunks)
    print(f"Truncated Text: {truncated_text}")

    # Repack text chunks to maximize the context window
    repacked_text = helper.repack(prompt, text_chunks)
    print(f"Repacked Text: {repacked_text}")


if __name__ == "__main__":
    main()
