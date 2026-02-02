from pathlib import Path
import json
from typing import List, Dict, Any, Optional

class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.results_file = self.checkpoint_dir / "results.json"
        self.progress_file = self.checkpoint_dir / "progress.json"

    def save_progress(self, processed_chunks: int, total_chunks: int):
        data = {
            "processed_chunks": processed_chunks,
            "total_chunks": total_chunks,
            "last_updated": str(Path().absolute())
        }
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_partial_results(self, results: List[Dict[str, Any]]):
        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def load_progress(self) -> Optional[Dict]:
        if not self.progress_file.exists():
            return None
        with open(self.progress_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_results(self) -> List[Dict[str, Any]]:
        if not self.results_file.exists():
            return []
        with open(self.results_file, "r", encoding="utf-8") as f:
            return json.load(f)
