import os
import re
import logging
from typing import List, Sequence
from autogen_core.code_executor import CodeBlock, CodeExecutor
from autogen_agentchat.agents import CodeExecutorAgent
from jet.logger import logger


class CustomCodeExecutorAgent(CodeExecutorAgent):
    def __init__(
        self,
        name: str,
        code_executor: CodeExecutor,
        *,
        description: str = "A computer terminal that performs no other action than running Python scripts (provided to it quoted in ```python code blocks), or sh shell scripts (provided to it quoted in ```sh code blocks).",
        sources: Sequence[str] | None = None,
    ) -> None:
        super().__init__(name=name, description=description,
                         code_executor=code_executor, sources=sources)
        self._test_code = ""
        with open(f"{os.path.dirname(__file__)}/test.txt", "rt") as fh:
            self._test_code = fh.read()

    def _extract_markdown_code_blocks(self, markdown_text: str) -> List[CodeBlock]:
        code_blocks = super()._extract_markdown_code_blocks(markdown_text)
        new_blocks: List[CodeBlock] = []
        for block in code_blocks:
            if block.language and block.language.lower() == "python":
                # Extract the function definition, including imports, docstring, and body
                function_match = re.match(
                    r"(from typing import List\n+def has_close_elements\(.*?-> bool:\n(?:\s*?\"{3}.*?\"{3}\n)?(?:\s*.*?\n)+?)(?=\n*(?:from|\Z))",
                    block.code,
                    re.DOTALL
                )
                if function_match:
                    code_content = function_match.group(1).rstrip()
                    # Combine test code, test runner, and function definition
                    code_content = (
                        self._test_code
                        + """
def run_tests(candidate):
    try:
        check(candidate)
        print("ALL TESTS PASSED !#!#")
        print("TERMINATE")
    except AssertionError:
        print("SOME TESTS FAILED - TRY AGAIN !#!#")
"""
                        + code_content
                        + """
run_tests(has_close_elements)
"""
                    )
                    logger.debug("Matched function definition: %s",
                                 function_match.group(1))
                    logger.debug("Constructed code block: %s", code_content)
                    new_blocks.append(
                        CodeBlock(code=code_content, language=block.language))
                else:
                    logger.warning(
                        "No valid function definition found in code block: %s", block.code)
            else:
                new_blocks.append(block)  # Non-Python blocks are unchanged
        return new_blocks
