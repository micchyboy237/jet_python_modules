"""
SpeakerResetHandler
===================
Makes an HTTP POST request to the live-subtitles server's /speakers/reset
endpoint. Designed to be called from the UI clear action so that speaker
diarization state is reset alongside the segment store and WS queue.

Usage
-----
    handler = SpeakerResetHandler()
    # ... later, on clear:
    handler.reset()
"""

import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

from rich.console import Console

console = Console()

# Default timeout in seconds for the HTTP request.
_REQUEST_TIMEOUT_S = 5


class SpeakerResetHandler:
    """
    Lightweight handler that POSTs to /speakers/reset on the live-subtitles server.

    The host is resolved once at init time via the LOCAL_LIVE_SUBTITLES_HOST
    environment variable so it can be overridden in different environments
    without touching code.
    """

    def __init__(self, timeout: float = _REQUEST_TIMEOUT_S) -> None:
        self._host: str = os.getenv("LOCAL_LIVE_SUBTITLES_HOST", "localhost:8000")
        self._url: str = f"http://{self._host}/speakers/reset"
        self._timeout: float = timeout

    def reset(self) -> bool:
        """
        Call the /speakers/reset endpoint.

        Returns True on success (HTTP 200 with success=True),
        False on any failure (network error, non-200, missing success flag).

        Errors are logged but never raised — this is fire-and-forget from the
        perspective of the UI clear action.
        """
        try:
            req = Request(
                self._url,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=self._timeout) as resp:
                if resp.status != 200:
                    console.print(
                        f"[yellow][SpeakerReset][/yellow] "
                        f"Unexpected status {resp.status} from {self._url}"
                    )
                    return False
                body = json.loads(resp.read().decode("utf-8"))
                success = body.get("success", False)
                message = body.get("message", "")
                if success:
                    console.print(
                        f"[green][SpeakerReset][/green] {message or 'Speaker state reset'}"
                    )
                else:
                    console.print(
                        f"[yellow][SpeakerReset][/yellow] "
                        f"Server reported failure: {message or 'unknown'}"
                    )
                return bool(success)
        except URLError as exc:
            console.print(
                f"[red][SpeakerReset][/red] Network error calling {self._url}: {exc}"
            )
            return False
        except json.JSONDecodeError as exc:
            console.print(
                f"[red][SpeakerReset][/red] Invalid JSON response from {self._url}: {exc}"
            )
            return False
        except Exception as exc:
            console.print(
                f"[red][SpeakerReset][/red] Unexpected error: {type(exc).__name__}: {exc}"
            )
            return False
