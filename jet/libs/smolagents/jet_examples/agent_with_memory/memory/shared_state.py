import json
import os
from typing import Any


class SharedState:
    def __init__(self, file_path: str = "agent_shared_state.json"):
        self.file_path = file_path
        self.data: dict = {}
        self.load()

    def load(self) -> None:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception as e:
                print(f"Warning: could not load shared state: {e}")

    def save(self) -> None:
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: could not save shared state: {e}")

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.save()

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def append_to_list(self, key: str, item: Any) -> None:
        lst = self.get(key, [])
        if not isinstance(lst, list):
            lst = []
        lst.append(item)
        self.set(key, lst)

    def __repr__(self) -> str:
        return f"SharedState(keys={list(self.data.keys())})"


# Global instance
shared_state = SharedState()
