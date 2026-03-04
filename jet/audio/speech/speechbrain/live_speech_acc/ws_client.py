# ws_client.py

import asyncio
import base64
import json
import queue
from typing import Any, Dict

import websockets
from rich.console import Console

console = Console()


class PersistentWSClient:
    """Manages a single persistent WebSocket connection for sending utterances."""

    def __init__(self, ws_uri: str):
        self.ws_uri = ws_uri
        self.send_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.task: asyncio.Task | None = None

    def start(self):
        """Start background sender task in a new event loop."""
        self.loop = asyncio.new_event_loop()

        def run_loop():
            asyncio.set_event_loop(self.loop)
            self.task = self.loop.create_task(self._sender_loop())
            self.loop.run_forever()

        import threading

        threading.Thread(target=run_loop, daemon=True).start()

    async def _sender_loop(self):
        while True:
            try:
                async with websockets.connect(self.ws_uri) as ws:
                    console.print("[bold green]Persistent WS connected[/bold green]")
                    while True:
                        payload = self.send_queue.get()
                        if payload is None:  # sentinel for shutdown
                            break
                        await ws.send(json.dumps(payload))
                        try:
                            response = await asyncio.wait_for(ws.recv(), timeout=15.0)
                            resp_data = json.loads(response)
                            console.print(
                                f"[bold magenta]Received translation: {resp_data.get('en_text')}[/bold magenta]"
                            )
                            # Optionally: callback or pub/sub to display
                        except asyncio.TimeoutError:
                            console.print("[yellow]Translation timeout[/yellow]")
            except Exception as e:
                console.print(f"[red]WS error: {e}. Reconnecting in 3s...[/red]")
                await asyncio.sleep(3.0)

    def send_audio(
        self,
        audio_bytes: bytearray,
        client_id: str,
        utterance_id: str,
        segment_num: int,
    ):
        """Thread-safe: queue audio for sending."""
        payload = {
            "type": "audio",
            "audio_bytes": base64.b64encode(audio_bytes).decode("utf-8"),
            "client_id": client_id,
            "utterance_id": utterance_id,
            "segment_num": segment_num,
        }
        self.send_queue.put(payload)
