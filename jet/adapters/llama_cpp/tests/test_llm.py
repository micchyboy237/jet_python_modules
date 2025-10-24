import pytest
from pydantic import BaseModel
from typing import List
from jet.adapters.llama_cpp.llm import LlamacppLLM, ChatMessage


class TestLlamacppLLMFunctional:
    """
    Functional tests using real llama-server.
    Ensure server is running at http://shawn-pc.local:8080/v1 with model loaded.
    """

    @pytest.fixture(scope="class")
    def llm(self):
        """Initialize LLM client pointing to local server."""
        return LlamacppLLM(
            model="qwen3-instruct-2507:4b",
            base_url="http://shawn-pc.local:8080/v1",
            api_key="sk-1234",
            verbose=True
        )

    # === Sync Chat (non-stream) ===
    def test_chat_sync(self, llm):
        # Given
        messages: List[ChatMessage] = [{"role": "user", "content": "Say 'Hello, world!' in one sentence."}]

        # When
        result = llm.chat(messages, temperature=0.0)

        # Then
        expected = "Hello, world!"
        assert expected.lower() in result.lower(), f"Expected '{expected}', got '{result}'"

    # === Sync Chat (stream) ===
    def test_chat_sync_stream(self, llm):
        # Given
        messages: List[ChatMessage] = [{"role": "user", "content": "Say 'Hello, world!' in one sentence."}]

        # When
        contents = list(llm.chat(messages, temperature=0.0, stream=True))
        result = "".join(contents)

        # Then
        expected = "Hello, world!"
        assert expected.lower() in result.lower(), f"Expected '{expected}', got '{result}'"

    # === Sync Chat (stream) ===
    def test_chat_stream(self, llm):
        # Given
        messages: List[ChatMessage] = [{"role": "user", "content": "Say: A B C"}]

        # When
        chunks = list(llm.chat_stream(messages, temperature=0.0))

        # Then
        full_text = "".join(c.choices[0].delta.content or "" for c in chunks)
        assert "A" in full_text and "B" in full_text and "C" in full_text

    # === Async Chat (stream) ===
    @pytest.mark.asyncio
    async def test_achat_stream(self, llm):
        # Given
        messages: List[ChatMessage] = [{"role": "user", "content": "Say: A B C"}]
        # When
        full_text = ""
        async for chunk in llm.achat_stream(messages, temperature=0.0):
            delta = chunk.choices[0].delta.content or ""
            full_text += delta
        # Then
        assert "A" in full_text and "B" in full_text and "C" in full_text

    # === Sync Completion ===
    def test_generate_sync(self, llm):
        # Given
        prompt = "The capital of France is"

        # When
        result = llm.generate(prompt, temperature=0.0)

        # Then
        expected = "Paris"
        assert expected in result, f"Expected '{expected}' in response, got '{result}'"

    # === Sync Completion (stream) ===
    def test_generate_sync_stream(self, llm):
        # Given
        prompt = "Complete the sequence with up to 3 more numbers: 1, 2, 3, "
        # When
        stream = llm.generate(prompt, temperature=0.0, stream=True)
        contents = list(stream)
        result = "".join(contents).strip()
        # Then
        expected_numbers = {"4", "5", "6"}
        found = {word.strip(",").strip() for word in result.split() if word.replace(",", "").strip().isdigit()}
        assert expected_numbers.issubset(found), \
            f"Expected numbers {expected_numbers} in stream, got '{result}'"

    # === Sync Tools ===
    def test_chat_with_tools(self, llm):
        # Given
        messages: List[ChatMessage] = [{"role": "user", "content": "What is 7 plus 8?"}]

        def add(a: int, b: int) -> int:
            return a + b

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]
        available_functions = {"add": add}

        # When
        result = llm.chat_with_tools(messages, tools, available_functions, temperature=0.0)

        # Then
        expected = "15"
        assert expected in result, f"Expected '{expected}' in final response, got '{result}'"

    # === Structured Output ===
    def test_chat_structured(self, llm):
        # Given
        class Friend(BaseModel):
            name: str
            age: int

        class FriendList(BaseModel):
            friends: List[Friend]

        messages: List[ChatMessage] = [
            {
                "role": "user",
                "content": "I have two friends: Alice is 25, Bob is 30. Return JSON list."
            }
        ]

        # When
        result = llm.chat_structured(messages, FriendList, temperature=0.0)

        # Then
        expected_names = {"Alice", "Bob"}
        result_names = {f.name for f in result.friends}
        assert expected_names == result_names, f"Expected names {expected_names}, got {result_names}"
        assert all(f.age > 0 for f in result.friends)

    # === Async Chat ===
    @pytest.mark.asyncio
    async def test_achat(self, llm):
        # Given
        messages: List[ChatMessage] = [{"role": "user", "content": "Respond with 'Async works!'"}]

        # When
        result = await llm.achat(messages, temperature=0.0)

        # Then
        expected = "Async works!"
        assert expected.lower() in result.lower(), f"Expected '{expected}', got '{result}'"

    # === Async Completion ===
    @pytest.mark.asyncio
    async def test_agenerate(self, llm):
        # Given
        prompt = "Complete: 2 + 2 ="

        # When
        result = await llm.agenerate(prompt, temperature=0.0)

        # Then
        expected = "4"
        assert expected in result.strip(), f"Expected '{expected}', got '{result}'"

    # === Async Tools ===
    @pytest.mark.asyncio
    async def test_achat_with_tools(self, llm):
        # Given
        messages: List[ChatMessage] = [{"role": "user", "content": "What is 10 minus 4?"}]

        def subtract(a: int, b: int) -> int:
            return a - b

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "subtract",
                    "description": "Subtract two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]
        available_functions = {"subtract": subtract}

        # When
        result = await llm.achat_with_tools(messages, tools, available_functions, temperature=0.0)

        # Then
        expected = "6"
        assert expected in result, f"Expected '{expected}' in final response, got '{result}'"

    # === Async Structured Output ===
    @pytest.mark.asyncio
    async def test_achat_structured(self, llm):
        # Given
        class Pet(BaseModel):
            name: str
            species: str
            age: int

        class PetList(BaseModel):
            pets: List[Pet]

        messages: List[ChatMessage] = [
            {
                "role": "user",
                "content": "My pets are: Max the dog (5 years), Luna the cat (3 years). Return JSON list."
            }
        ]

        # When
        result = await llm.achat_structured(messages, PetList, temperature=0.0)

        # Then
        expected_names = {"Max", "Luna"}
        result_names = {p.name for p in result.pets}
        assert expected_names == result_names, f"Expected names {expected_names}, got {result_names}"
        assert all(p.age > 0 for p in result.pets)
        species = {p.species.lower() for p in result.pets}
        assert "dog" in species and "cat" in species
