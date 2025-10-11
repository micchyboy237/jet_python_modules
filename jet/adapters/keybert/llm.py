from keybert import KeyLLM as BaseKeyLLM
# from keybert.llm._base import BaseLLM

class KeyLLM(BaseKeyLLM):
    def __init__(self, llm):
        super().__init__(
            llm=llm
        )
