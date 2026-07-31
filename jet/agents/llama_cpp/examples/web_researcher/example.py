import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents.manager import create_web_researcher

# Create the researcher
researcher = create_web_researcher(
    llm_model="qwen3.5-uncensored:2b",
    base_url="http://localhost:8080",
    max_context_length=4096,
    verbose=True,
)

# Run a research query
result = researcher.run("What are the latest advancements in renewable energy storage?")
stats = researcher.get_token_stats()

print(result)
print(f"Total tokens used: {stats['total_tokens']}")
