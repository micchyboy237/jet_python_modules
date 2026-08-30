# jet_python_modules/jet/agents/llama_cpp/live_rag_search/memory.py

import json
from dataclasses import dataclass, field


@dataclass
class AccumulatedMemory:
    facts: dict[str, dict] = field(default_factory=dict)
    total_fact_count: int = 0

    def add_facts(self, new_entities: dict[str, dict], limit: int) -> int:
        added = 0
        for entity_id, entity_data in new_entities.items():
            if self.total_fact_count >= limit:
                break

            # Defensive check: Ensure entity_data is a dict
            if not isinstance(entity_data, dict):
                if isinstance(entity_data, str):
                    entity_data = {"value": entity_data}
                else:
                    continue

            if entity_id not in self.facts:
                self.facts[entity_id] = dict(entity_data)
                self.total_fact_count += len(entity_data)
                added += len(entity_data)
            else:
                for k, v in entity_data.items():
                    if k not in self.facts[entity_id] and self.total_fact_count < limit:
                        self.facts[entity_id][k] = v
                        self.total_fact_count += 1
                        added += 1
        return added

    def get_entity_ids(self) -> set[str]:
        return set(self.facts.keys())

    def to_context_string(self) -> str:
        if not self.facts:
            return "(No accumulated facts yet)"
        return json.dumps(self.facts, ensure_ascii=False, indent=2)

    @property
    def is_empty(self) -> bool:
        return self.total_fact_count == 0
