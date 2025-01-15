from copy import deepcopy
from typing import Any, Callable, Optional
from llama_index.core.prompts.base import BasePromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.prompts.utils import get_template_vars, format_string
from llama_index.core.base.llms.generic_utils import prompt_to_messages
from llama_index.core.types import BaseOutputParser
from pydantic.fields import Field


class MetadataPromptTemplate(BasePromptTemplate):
    template: str = Field(
        description="The custom template.",
    )

    def __init__(
        self,
        template: str,
        prompt_type: str = PromptType.CUSTOM,
        output_parser: Optional[BaseOutputParser] = None,
        metadata: Optional[dict[str, Any]] = None,
        template_var_mappings: Optional[dict[str, Any]] = None,
        function_mappings: Optional[dict[str, Callable]] = None,
        **kwargs: Any,
    ) -> None:
        if metadata is None:
            metadata = {}
        metadata["prompt_type"] = prompt_type

        template_vars = get_template_vars(template)

        super().__init__(
            template=template,
            template_vars=template_vars,
            kwargs=kwargs,
            metadata=metadata,
            output_parser=output_parser,
            template_var_mappings=template_var_mappings,
            function_mappings=function_mappings,
        )

    def partial_format(self, **kwargs: Any) -> "MetadataPromptTemplate":
        """Partially format the custom prompt."""
        prompt = deepcopy(self)
        prompt.kwargs.update(kwargs)
        return prompt

    def format(self, llm: Optional[BaseLLM] = None, **kwargs: Any) -> str:
        """Format the custom prompt into a string."""
        del llm  # unused
        all_kwargs = {**self.kwargs, **kwargs}
        mapped_all_kwargs = self._map_all_vars(all_kwargs)
        return format_string(self.template, **mapped_all_kwargs)

    def format_messages(self, llm: Optional[BaseLLM] = None, **kwargs: Any) -> list[ChatMessage]:
        del llm  # unused
        """Format the custom prompt into a list of chat messages."""
        prompt = self.format(**kwargs)
        return prompt_to_messages(prompt)

    def get_template(self, llm: Optional[BaseLLM] = None) -> str:
        return self.template


# Sample usage of MetadataPromptTemplate
def main():
    from jet.logger import logger
    template = (
        "Context information is below.\n"
        "---------------------\n{context_str}\n---------------------\n"
        "Given the context information and not prior knowledge, answer the query.\n"
        "Please also write the answer in the style of {tone_name}.\n"
        "Query: {query_str}\nAnswer: "
    )

    # Creating an instance of the MetadataPromptTemplate
    prompt_template = MetadataPromptTemplate(
        template=template,
        context_str="I am a developer working on a React Native app.",
        tone_name="friendly",
        query_str="Tell me about yourself."
    )

    # Populating the required template variables
    formatted_prompt = prompt_template.format()

    logger.newline()
    logger.info("formatted_prompt")
    logger.success(formatted_prompt)


if __name__ == "__main__":
    main()
