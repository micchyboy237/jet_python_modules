# jet.audio.audio_waveform.helpers.subtitle_entry

import json
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class SubtitleEntry:
    @staticmethod
    def _format_time(s: float) -> str:
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = s % 60
        ms = int((sec - int(sec)) * 1000)
        return f"{h:02d}:{m:02d}:{int(sec):02d},{ms:03d}"

    def __init__(self, output_path: Path | None = None):
        self.entries: List[dict] = []
        self.by_uuid: Dict[str, dict] = {}
        self.output_path: Path | None = output_path
        self._lock = threading.Lock()

    def add_pending(
        self,
        uuid_str: str,
        start_sec: float,
        end_sec: float,
        segment_id: int,
        started_at: str,
        segment_dir: Path | None = None,
        trigger_reason: str | None = None,
    ):
        with self._lock:
            entry = {
                "uuid": uuid_str,
                "segment_id": segment_id,
                "index": len(self.entries) + 1 + len(self.by_uuid),
                "start": start_sec,
                "end": end_sec,
                "ja": "",
                "en": "",
                "started_at": started_at,
                "received_at": None,
                "final": False,
                "trigger_reason": trigger_reason,
                "segment_dir": segment_dir,  # will be used for per-segment other_results.json
            }
            self.by_uuid[uuid_str] = entry
            # segment_dir is now stored only inside the entry dict (no duplicate mapping)

    def update(self, uuid_str: str, ja: str, en: str, others: dict | None = None):
        with self._lock:
            if uuid_str not in self.by_uuid:
                print(f"[Subtitle] Warning: received unknown uuid {uuid_str}")
                return

            e = self.by_uuid[uuid_str]
            e["ja"] = ja.strip()
            e["en"] = en.strip()
            e["received_at"] = datetime.utcnow().isoformat()
            e["final"] = True

            if others:
                # Ensure we always have a response dict for saving
                response = {"ja": ja.strip(), "en": en.strip(), **(others or {})}
                # Store metadata directly in the entry so live preview can read it
                e.update(others)
                segment_dir = e.get("segment_dir")
                self._write_segment_response(segment_dir, response)

            if e["ja"] or e["en"]:
                self.entries.append(e)
                # Always write segment SRT/response even if some fields were missing
                segment_dir = e.get("segment_dir")
                if segment_dir:
                    self._write_segment_srt(e["uuid"])
                del self.by_uuid[uuid_str]
                self.entries.sort(key=lambda x: x["start"])
                self._write_global_srt()

    def _write_segment_response(self, segment_dir: Path | None, response: dict):
        """Save the full response dictionary (including ja and en) as response.json
        inside the segment_dir."""
        if not self.output_path or not segment_dir:
            return

        try:
            json_path = segment_dir / "response.json"  # now always receives a full dict
            json_path.write_text(
                json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[JSON] Segment response.json saved successfully: {json_path}")

        except Exception as e:
            print(f"[JSON] Failed writing response.json: {e}")

    def _write_global_srt(self):
        if not self.output_path:
            return

        try:
            self.output_path.write_text(self.to_srt(), encoding="utf-8")
            print(f"[SRT] Global subtitles updated successfully: {self.output_path}")
        except Exception as e:
            print(f"[SRT] Failed writing global SRT: {e}")

    def _write_segment_srt(self, uuid_str: str):
        try:
            entry = next((e for e in self.entries if e["uuid"] == uuid_str), None)
            if not entry:
                return
            segment_dir = entry.get("segment_dir")
            if not segment_dir:
                return

            path = segment_dir / "subtitles.srt"

            start = self._format_time(entry["start"])
            end = self._format_time(entry["end"])
            text = f"{entry['ja']}\n{entry['en']}".strip()

            content = "\n".join(["1", f"{start} --> {end}", text, ""])

            path.write_text(content, encoding="utf-8")
            print(f"[SRT] Segment subtitles updated successfully: {path}")

        except Exception as e:
            print(f"[SRT] Failed writing segment SRT: {e}")

    def to_srt(self) -> str:
        lines = []
        for i, e in enumerate(self.entries, 1):
            start = self._format_time(e["start"])
            end = self._format_time(e["end"])
            text = f"{e['ja']}\n{e['en']}".strip()  # empty string if no transcription
            lines.extend([str(i), f"{start} --> {end}", text, ""])
        return "\n".join(lines)

    def clear(self) -> None:
        """
        Clear all stored subtitle entries and pending data.
        Resets the accumulator to its initial empty state.
        """
        with self._lock:
            self.entries.clear()
            self.by_uuid.clear()
            # uuid_to_segment_dir removed entirely

        shutil.rmtree(self.output_path.parent, ignore_errors=True)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        print("[SubtitleEntry] All entries, pending and saved data are cleared")
