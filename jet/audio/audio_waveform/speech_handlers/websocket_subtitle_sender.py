# jet.audio.audio_waveform.speech_handlers.websocket_subtitle_sender

import asyncio
import json
import os
import threading
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
import soundfile as sf
from jet.audio.audio_waveform.helpers.subtitle_entry import SubtitleEntry
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


class WebsocketSubtitleSender(SpeechSegmentHandler):
    def __init__(
        self,
        accumulator: SubtitleEntry,
        ws_url: str | None = None,
        reconnect_attempts: int = 100,
        reconnect_delay: float = 2.0,
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
        self._stop_event = threading.Event()
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

        while not self._stop_event.is_set():
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

            if self._stop_event.is_set():
                print("[WS] Stop requested — exiting reconnect loop")
                break

            attempt += 1
            if attempt >= self.reconnect_attempts:
                print(
                    f"[WS] Max reconnection attempts ({self.reconnect_attempts}) reached. Giving up."
                )
                break

            delay = self.reconnect_delay
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

                    ja = header["transcription_ja"]
                    en = header["translation_en"]
                    others = {
                        "transcribed_duration_sec": header["transcribed_duration_sec"],
                        "transcribed_duration_pctg": header[
                            "transcribed_duration_pctg"
                        ],
                        "coverage_label": header["coverage_label"],
                        "phrase_segments": header["phrase_segments"],
                    }

                    print(
                        f"[WS ←] {uid[:8]}…  ja: {ja[:50]}{'…' if len(ja) > 50 else ''}"
                    )
                    if en:
                        print(
                            f"               en: {en[:50]}{'…' if len(en) > 50 else ''}"
                        )

                    self.accumulator.update(uid, ja, en, others)

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
            event.started_at,
            segment_dir=event.segment_dir,
            trigger_reason=event.trigger_reason,
        )

        if self.debug_dir:
            sf.write(
                os.path.join(
                    self.debug_dir, f"seg_{event.segment_id:03d}_{seg_uuid[:8]}.wav"
                ),
                event.audio,
                16000,
            )

        # Convert float32 [-1,1] → int16 [-32768, 32767] before sending
        audio_int16 = np.int16(event.audio * 32767.0).tobytes()

        header = {
            "uuid": seg_uuid,
            "start_sec": event.start_time_sec,
            "end_sec": event.end_time_sec,
            "duration_sec": event.duration_sec,
            "sample_rate": 16000,
            "format": "int16le",  # changed from float32le
            "channels": 1,
            "language": "ja",
            "vad_reason": event.trigger_reason,
            "forced": event.forced_split,
            "segment_rms": event.segment_rms,
            "loudness": event.loudness,
            "has_sound": event.has_sound,
            "started_at": event.started_at,
        }
        json_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

        payload = json_bytes + b"\x00" + audio_int16

        async def send_payload():
            if self.ws is None:
                print("[WS →] Cannot send — no active connection")
                return
            try:
                await self.ws.send(payload)
                print(
                    f"[WS →] sent seg {event.segment_id}  uuid={seg_uuid[:8]}…  "
                    f"{len(audio_int16) / 1024:.1f} KiB"
                )
            except ConnectionClosed:
                print("[WS →] Send failed — connection closed (will retry later)")
            except Exception as e:
                print(f"[WS →] Send error: {e}")

        self.loop.call_soon_threadsafe(lambda: self.loop.create_task(send_payload()))

    def close(self):
        self._stop_event.set()

        if self.ws is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
                future.result(timeout=3.0)
            except Exception as e:
                print(f"[WS] Error during close: {e}")

        # stop event loop
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

        # wait for thread to exit
        if self._loop_thread.is_alive():
            self._loop_thread.join(timeout=3.0)

        print("[WS] Client shutdown requested")
