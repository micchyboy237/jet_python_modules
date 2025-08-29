# .pyenv/versions/3.12.9/lib/python3.12/site-packages/llama_index/core/question_gen/llm_generators.py
from typing import List, Optional, Sequence, cast
from llama_index.core.llms.llm import LLM
from llama_index.core.output_parsers.base import StructuredOutput
from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.question_gen.output_parser import SubQuestionOutputParser
from llama_index.core.question_gen.prompts import (
    DEFAULT_SUB_QUESTION_PROMPT_TMPL,
    build_tools_text,
)
from llama_index.core.question_gen.types import BaseQuestionGenerator, SubQuestion
from llama_index.core.schema import QueryBundle
from llama_index.core.settings import Settings
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.types import BaseOutputParser

# Custom prompt to ensure JSON output
CUSTOM_SUB_QUESTION_PROMPT_TMPL = """
You are an expert at breaking down complex queries into sub-questions. Given a query and a set of tools, generate a list of sub-questions that can be answered using the provided tools. Return the response as a JSON array of objects, where each object has a 'sub_question' field (the sub-question text) and a 'tool_name' field (the name of the tool to use). Ensure the output is valid JSON.

Tools:
{tools_str}

Query:
{query_str}

Example output:
[
    {{"sub_question": "What is the risk factor for Uber in 2022?", "tool_name": "vector_index_2022"}},
    {{"sub_question": "What is the risk factor for Uber in 2021?", "tool_name": "vector_index_2021"}}
]
"""


class LLMQuestionGenerator(BaseQuestionGenerator):
    def __init__(
        self,
        llm: LLM,
        prompt: BasePromptTemplate,
    ) -> None:
        self._llm = llm
        self._prompt = prompt
        if self._prompt.output_parser is None:
            raise ValueError("Prompt should have output parser.")

    @classmethod
    def from_defaults(
        cls,
        llm: Optional[LLM] = None,
        prompt_template_str: Optional[str] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> "LLMQuestionGenerator":
        llm = llm or Settings.llm
        prompt_template_str = prompt_template_str or CUSTOM_SUB_QUESTION_PROMPT_TMPL
        output_parser = output_parser or SubQuestionOutputParser()
        prompt = PromptTemplate(
            template=prompt_template_str,
            output_parser=output_parser,
            prompt_type=PromptType.SUB_QUESTION,
        )
        return cls(llm, prompt)

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"question_gen_prompt": self._prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "question_gen_prompt" in prompts:
            output_parser = prompts["question_gen_prompt"].output_parser
            if output_parser is None:
                output_parser = SubQuestionOutputParser()
            self._prompt = PromptTemplate(
                prompts["question_gen_prompt"].get_template(llm=self._llm),
                output_parser=output_parser,
            )

    def generate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        tools_str = build_tools_text(tools)
        query_str = query.query_str
        prediction = self._llm.predict(
            prompt=self._prompt,
            tools_str=tools_str,
            query_str=query_str,
            format="json",  # Enforce JSON mode
        )
        assert self._prompt.output_parser is not None
        parse = self._prompt.output_parser.parse(prediction)
        parse = cast(StructuredOutput, parse)
        return parse.parsed_output

    async def agenerate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        tools_str = build_tools_text(tools)
        query_str = query.query_str
        prediction = await self._llm.apredict(
            prompt=self._prompt,
            tools_str=tools_str,
            query_str=query_str,
            format="json",  # Enforce JSON mode
        )
        assert self._prompt.output_parser is not None
        parse = self._prompt.output_parser.parse(prediction)
        parse = cast(StructuredOutput, parse)
        return parse.parsed_output
