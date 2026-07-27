"""
File-based artifact store for orchestrator <-> subagent handoffs.

Instead of passing full conversational history between agents (which
re-inflates everyone's context), subagents write their complete output to
a JSON file on disk and pass back only a small reference + a compressed
summary. The orchestrator (and any other subagent that needs the detail
later) can open the file directly without it ever occupying the live
context window.

This is the "artifact pattern": write once, reference many times, only
pull full detail into context when something specifically needs it.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class Artifact:
    id: str
    kind: str          # e.g. "subagent_report", "plan", "final_answer"
    agent_id: str
    created_at: float
    summary: str        # short, goes back into orchestrator context
    content: str         # full detail, stays on disk
    metadata: dict


class ArtifactStore:
    def __init__(self, directory: str = "artifacts"):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        kind: str,
        agent_id: str,
        summary: str,
        content: str,
        metadata: dict | None = None,
    ) -> Artifact:
        artifact = Artifact(
            id=str(uuid.uuid4())[:8],
            kind=kind,
            agent_id=agent_id,
            created_at=time.time(),
            summary=summary,
            content=content,
            metadata=metadata or {},
        )
        path = self.dir / f"{artifact.id}.json"
        path.write_text(json.dumps(asdict(artifact), indent=2))
        return artifact

    def read(self, artifact_id: str) -> Artifact:
        path = self.dir / f"{artifact_id}.json"
        data = json.loads(path.read_text())
        return Artifact(**data)

    def read_content(self, artifact_id: str) -> str:
        """Convenience: pull just the full content, e.g. when a later
        subagent or the final synthesis step actually needs the detail."""
        return self.read(artifact_id).content

    def list_by_kind(self, kind: str) -> list[Artifact]:
        out = []
        for path in sorted(self.dir.glob("*.json")):
            data = json.loads(path.read_text())
            if data["kind"] == kind:
                out.append(Artifact(**data))
        return out

    def reference(self, artifact: Artifact) -> str:
        """What actually goes back into the orchestrator's live context:
        a small pointer + summary, never the full content."""
        return (
            f"[artifact:{artifact.id}] ({artifact.kind}, by {artifact.agent_id})\n"
            f"{artifact.summary}"
        )
