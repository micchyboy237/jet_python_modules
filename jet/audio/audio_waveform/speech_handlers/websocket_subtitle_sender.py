# jet.audio.audio_waveform.speech_handlers.websocket_subtitle_sender

import asyncio
import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import soundfile as sf
from jet.audio.audio_waveform.speech_events import (
    SpeechSegmentEndEvent,
    SpeechSegmentStartEvent,
)
from jet.audio.audio_waveform.speech_handlers.base import SpeechSegmentHandler

# === UPDATED IMPORTS FOR WEBSOCKETS 16.0+ ===
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
)


class SubtitleEntry:
    def __init__(self, output_path: Path | None = None):
        self.entries: List[dict] = []
        self.by_uuid: Dict[str, dict] = {}
        self.output_path: Path | None = output_path
        self.uuid_to_segment_dir: Dict[str, Path] = {}

    def add_pending(
        self,
        uuid_str: str,
        start_sec: float,
        end_sec: float,
        segment_id: int,
        segment_dir: Path | None = None,
    ):
        entry = {
            "uuid": uuid_str,
            "index": len(self.entries) + 1 + len(self.by_uuid),
            "start": start_sec,
            "end": end_sec,
            "ja": "",
            "en": "",
            "segment_id": segment_id,
            "received_at": None,
            "final": False,
        }

        self.by_uuid[uuid_str] = entry

        if segment_dir:
            self.uuid_to_segment_dir[uuid_str] = segment_dir

    def update(self, uuid_str: str, ja: str, en: str):
        if uuid_str not in self.by_uuid:
            print(f"[Subtitle] Warning: received unknown uuid {uuid_str}")
            return

        e = self.by_uuid[uuid_str]
        e["ja"] = ja.strip()
        e["en"] = en.strip()
        e["received_at"] = datetime.utcnow().isoformat()
        e["final"] = True

        if e["ja"] or e["en"]:
            self.entries.append(e)
            del self.by_uuid[uuid_str]

            self.entries.sort(key=lambda x: x["start"])

            # ✅ NEW: write immediately
            self._write_global_srt()
            self._write_segment_srt(uuid_str)

    def _write_global_srt(self):
        if not self.output_path:
            return

        try:
            self.output_path.write_text(self.to_srt(), encoding="utf-8")
            print(f"[SRT] Global subtitles updated successfully: {self.output_path}")
        except Exception as e:
            print(f"[SRT] Failed writing global SRT: {e}")

    def _write_segment_srt(self, uuid_str: str):
        segment_dir = self.uuid_to_segment_dir.get(uuid_str)
        if not segment_dir:
            return

        try:
            path = segment_dir / "subtitles.srt"
            path.write_text(self.to_srt(), encoding="utf-8")
            print(f"[SRT] Segment subtitles updated successfully: {path}")
        except Exception as e:
            print(f"[SRT] Failed writing segment SRT: {e}")

    def to_srt(self) -> str:
        lines = []
        for i, e in enumerate(self.entries, 1):
            start = self._format_time(e["start"])
            end = self._format_time(e["end"])
            text = f"{e['ja']}\n{e['en']}".strip()
            if not text:
                text = "[no transcription]"
            lines.extend([str(i), f"{start} --> {end}", text, ""])
        return "\n".join(lines)

    @staticmethod
    def _format_time(s: float) -> str:
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = s % 60
        ms = int((sec - int(sec)) * 1000)
        return f"{h:02d}:{m:02d}:{int(sec):02d},{ms:03d}"


class WebsocketSubtitleSender(SpeechSegmentHandler):
    def __init__(
        self,
        accumulator: SubtitleEntry,
        ws_url: str | None = None,
        reconnect_attempts: int = 20,
        reconnect_delay: float = 1.0,
        debug_save_audio: bool = False,
        debug_dir: str = "debug_segments",
    ):
        self.ws_url = ws_url or os.getenv("LOCAL_WS_LIVE_SUBTITLES_URL")
        if not self.ws_url:
            raise ValueError("LOCAL_WS_LIVE_SUBTITLES_URL not set or empty")

        print(f"[WS init] Using URL: {self.ws_url!r}")

        self.accumulator = accumulator
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self.debug_dir = None
        if debug_save_audio:
            self.debug_dir = os.path.join(
                debug_dir, datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            os.makedirs(self.debug_dir, exist_ok=True)

        self.ws: Optional[ClientConnection] = None
        self.loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
        )
        self._loop_thread.start()

        self.loop.call_soon_threadsafe(
            lambda: self.loop.create_task(self._connection_manager())
        )

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _connection_manager(self):
        attempt = 0

        print(f"[WS] Connecting → {self.ws_url}")

        while True:
            try:
                async with connect(
                    self.ws_url,
                    max_size=None,  # unlimited (matches original intent)
                    compression=None,  # disable compression
                    ping_interval=30,
                    ping_timeout=30,
                    close_timeout=10,
                ) as ws:
                    self.ws = ws
                    attempt = 0  # reset on successful connection
                    print(f"[WS] Connected to {self.ws_url}")

                    # Run sender & receiver concurrently
                    await asyncio.gather(
                        self._receiver(),
                    )

            except (ConnectionClosedOK, ConnectionClosedError) as e:
                print(f"[WS] Connection closed cleanly: {e}")
            except OSError as e:
                print(f"[WS] Network error: {e}")
            except Exception as e:
                print(f"[WS] Unexpected error in connection: {type(e).__name__}: {e}")
                import traceback

                traceback.print_exc()

            self.ws = None

            attempt += 1
            if attempt >= self.reconnect_attempts:
                print(
                    f"[WS] Max reconnection attempts ({self.reconnect_attempts}) reached. Giving up."
                )
                break

            delay = self.reconnect_delay * (2 ** (attempt - 1))
            print(
                f"[WS] Reconnecting in {delay:.1f}s (attempt {attempt}/{self.reconnect_attempts})..."
            )
            await asyncio.sleep(delay)

        print("[WS] Connection manager shutting down")

    async def _receiver(self):
        if self.ws is None:
            return

        try:
            async for msg in self.ws:
                try:
                    header = None

                    # --- CASE 1: Binary message ---
                    if isinstance(msg, bytes):
                        if b"\x00" in msg:
                            # header + audio payload
                            header_bytes, _ = msg.split(b"\x00", 1)
                            header = json.loads(header_bytes.decode("utf-8"))
                        else:
                            # plain JSON bytes (current server behavior)
                            header = json.loads(msg.decode("utf-8"))

                    # --- CASE 2: Text message (fallback support) ---
                    elif isinstance(msg, str):
                        header = json.loads(msg)

                    else:
                        print(f"[WS] Unknown message type: {type(msg)}")
                        continue

                    # --- Validate header ---
                    if not isinstance(header, dict):
                        print("[WS] Invalid header format (not dict)")
                        continue

                    uid = header.get("uuid")
                    if not uid:
                        print("[WS] Missing uuid in message")
                        continue

                    ja = header.get("transcription_ja", "").strip()
                    en = header.get("translation_en", "").strip()

                    print(
                        f"[WS ←] {uid[:8]}…  ja: {ja[:50]}{'…' if len(ja) > 50 else ''}"
                    )
                    if en:
                        print(
                            f"               en: {en[:50]}{'…' if len(en) > 50 else ''}"
                        )

                    self.accumulator.update(uid, ja, en)

                except json.JSONDecodeError as e:
                    print(f"[WS] JSON decode error: {e}")
                except Exception as e:
                    print(f"[WS] Parse error: {e}")

        except ConnectionClosed:
            print("[WS receiver] Connection closed — will be handled by manager")
        except Exception as e:
            print(f"[WS receiver] Unexpected error: {e}")

    def on_segment_start(self, event: SpeechSegmentStartEvent) -> None:
        pass

    def on_segment_end(self, event: SpeechSegmentEndEvent) -> None:
        if event.audio.size == 0:
            return

        seg_uuid = str(uuid.uuid4())

        self.accumulator.add_pending(
            seg_uuid,
            event.start_time_sec,
            event.end_time_sec,
            event.segment_id,
            segment_dir=event.segment_dir,
        )

        if self.debug_dir:
            sf.write(
                os.path.join(
                    self.debug_dir, f"seg_{event.segment_id:03d}_{seg_uuid[:8]}.wav"
                ),
                event.audio,
                16000,
            )

        header = {
            "uuid": seg_uuid,
            "start_sec": round(event.start_time_sec, 3),
            "end_sec": round(event.end_time_sec, 3),
            "duration_sec": round(event.duration_sec, 3),
            "sample_rate": 16000,
            "format": "float32le",
            "channels": 1,
            "language": "ja",
            "vad_reason": event.trigger_reason,
            "forced": event.forced_split,
        }
        json_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        audio_bytes = event.audio.tobytes()

        payload = json_bytes + b"\x00" + audio_bytes

        async def send_payload():
            if self.ws is None:
                print("[WS →] Cannot send — no active connection")
                return
            try:
                await self.ws.send(payload)
                print(
                    f"[WS →] sent seg {event.segment_id}  uuid={seg_uuid[:8]}…  "
                    f"{len(audio_bytes) / 1024:.1f} KiB"
                )
            except ConnectionClosed:
                print("[WS →] Send failed — connection closed (will retry later)")
            except Exception as e:
                print(f"[WS →] Send error: {e}")

        self.loop.call_soon_threadsafe(lambda: self.loop.create_task(send_payload()))

    def close(self):
        if self.ws is not None:
            try:
                # Best-effort synchronous close
                future = asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
                future.result(timeout=3.0)
            except Exception as e:
                print(f"[WS] Error during close: {e}")
        print("[WS] Client shutdown requested")
