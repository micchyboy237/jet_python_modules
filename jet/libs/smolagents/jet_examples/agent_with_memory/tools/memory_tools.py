from memory.long_term import long_term_memory
from memory.shared_state import shared_state
from smolagents import Tool


class LongTermSaveTool(Tool):
    name = "long_term_save"
    description = (
        "Save concise, high-value facts, entities, goals, lessons or user "
        "preferences to long-term memory for future reuse."
    )
    inputs = {
        "content": {
            "type": "string",
            "description": "The fact or information to store in long-term memory.",
        },
        "step_number": {
            "type": "integer",
            "description": "Optional step number associated with the fact.",
            "optional": True,
        },
    }
    output_type = "string"

    def forward(self, content: str, step_number: int | None = None) -> str:
        if len(content) < 8:
            return "Fact too short – ignored."
        step_nr = step_number if step_number is not None else 0
        return long_term_memory.add_fact(content, step_nr)


class LongTermRecallTool(Tool):
    name = "long_term_recall"
    description = "Search long-term memory for facts relevant to the current task."
    inputs = {
        "query": {
            "type": "string",
            "description": "Search query describing what to recall.",
        },
        "n_results": {
            "type": "integer",
            "description": "Maximum number of results to return.",
            "optional": True,
        },
    }
    output_type = "string"

    def forward(self, query: str, n_results: int = 5) -> str:
        return long_term_memory.search(query, n_results)


class SharedStateUpdateTool(Tool):
    name = "shared_state_update"
    description = "Update a named value in the persistent shared state."
    inputs = {
        "key": {
            "type": "string",
            "description": "State key to update.",
        },
        "value": {
            "type": "string",
            "description": "Value to store for the given key.",
        },
    }
    output_type = "string"

    def forward(self, key: str, value: str) -> str:
        shared_state.set(key, value)
        return f"Shared state updated → {key} = {value}"


class SharedStateReadTool(Tool):
    name = "shared_state_read"
    description = "Read a named value from the shared persistent state."
    inputs = {
        "key": {
            "type": "string",
            "description": "State key to read.",
        }
    }
    output_type = "string"

    def forward(self, key: str) -> str:
        v = shared_state.get(key, "<not set>")
        return f"{key}: {v}"
