from jet.llm.mlx.base import MLX
from jet.logger import logger


class PromptGenerator:
    def __init__(self, query, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
        self.query = query.strip()
        self.model = model
        self._client = MLX(model)

    def _get_llm_response(self, prompt: str) -> str:
        """Get the LLM response for a given prompt using the MLX client."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert prompt generator. Follow the instructions to create a clear, "
                    "structured prompt for solving a query. Ensure the prompt is concise and includes "
                    "all necessary guidance for reasoning and output format."
                )
            },
            {
                "role": "user",
                "content": prompt
            },
        ]

        full_response = ""
        for response in self._client.stream_chat(
            messages=messages,
            model=self.model,
            max_tokens=500,
            temperature=0.8,
        ):
            if response["choices"]:
                content = response["choices"][0].get(
                    "message", {}).get("content", "")
                full_response += content
                logger.success(content, flush=True)

        logger.newline()

        return full_response.strip()

    def generate_cot_prompt(self):
        """Generate a Chain-of-Thought prompt using the MLX client."""
        instruction = (
            f"Create a Chain-of-Thought prompt for solving the following query: '{self.query}'. "
            "The prompt should instruct the solver to think step-by-step, show all reasoning and calculations clearly, "
            "use relevant formulas if applicable (e.g., area = length × width for geometric problems), "
            "and provide the final answer in the format: 'Final Answer: [answer]'."
        )
        return self._get_llm_response(instruction)

    def generate_tot_prompt(self):
        """Generate a Tree-of-Thoughts prompt using the MLX client."""
        instruction = (
            f"Create a Tree-of-Thoughts prompt for solving the following query: '{self.query}'. "
            "The prompt should instruct three experts to collaborate, with each proposing a step "
            "(e.g., calculate a value, explore an approach, or reason about a subproblem), "
            "evaluating it as 'correct,' 'promising,' or 'incorrect' based on accuracy and relevance, "
            "and continuing with promising steps. Include relevant formulas if applicable "
            "(e.g., area = length × width for geometric problems). "
            "Provide the final answer in the format: 'Final Answer: [answer]'."
        )
        return self._get_llm_response(instruction)

    def generate_prompts(self):
        """Generate CoT and ToT prompts using the MLX client."""
        cot_prompt = self.generate_cot_prompt()
        tot_prompt = self.generate_tot_prompt()
        return {"cot_prompt": cot_prompt, "tot_prompt": tot_prompt}
