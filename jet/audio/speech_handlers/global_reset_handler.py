"""
GlobalResetHandler
==================
Makes an HTTP POST request to the live-subtitles server's /global/reset
endpoint. Designed to be called from the UI clear action so that all
application state is reset alongside the segment store and WS queue.

Usage
-----
    handler = GlobalResetHandler()
    # ... later, on clear:
    handler.reset()
"""

import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

from rich.console import Console

console = Console()

_REQUEST_TIMEOUT_S = 5


class GlobalResetHandler:
    """
    Lightweight handler that POSTs to /global/reset on the live-subtitles server.
    The host is resolved once at init time via the LOCAL_LIVE_SUBTITLES_HOST
    environment variable so it can be overridden in different environments
    without touching code.
    """

    def __init__(self, timeout: float = _REQUEST_TIMEOUT_S) -> None:
        self._host: str = os.getenv("LOCAL_LIVE_SUBTITLES_HOST", "localhost:8000")
        self._url: str = f"http://{self._host}/global/reset"
        self._timeout: float = timeout

    def reset(self, reset_type: str = "full", debug: bool = False) -> bool:
        """
        Call the /global/reset endpoint.

        Args:
            reset_type: 'full' (default) or 'soft'
            debug: Enable detailed trace logging

        Returns:
            True on success (HTTP 200 with success=True),
            False on any failure (network error, non-200, missing success flag).

        Errors are logged but never raised — this is fire-and-forget from the
        perspective of the UI clear action.
        """
        import urllib.parse

        try:
            # Send as form data to match the server's Form(...) parameters
            data = urllib.parse.urlencode(
                {"reset_type": reset_type, "debug": str(debug).lower()}
            ).encode("utf-8")

            req = Request(
                self._url,
                method="POST",
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            with urlopen(req, timeout=self._timeout) as resp:
                if resp.status != 200:
                    console.print(
                        f"[yellow][GlobalReset][/yellow] "
                        f"Unexpected status {resp.status} from {self._url}"
                    )
                    return False
                body = json.loads(resp.read().decode("utf-8"))
                success = body.get("success", False)
                message = body.get("message", "")
                if success:
                    console.print(
                        f"[green][GlobalReset][/green] {message or 'Global reset successful'}"
                    )
                else:
                    console.print(
                        f"[yellow][GlobalReset][/yellow] "
                        f"Server reported failure: {message or 'unknown'}"
                    )
                return bool(success)
        except URLError as exc:
            console.print(
                f"[red][GlobalReset][/red] Network error calling {self._url}: {exc}"
            )
            return False
        except json.JSONDecodeError as exc:
            console.print(
                f"[red][GlobalReset][/red] Invalid JSON response from {self._url}: {exc}"
            )
            return False
        except Exception as exc:
            console.print(
                f"[red][GlobalReset][/red] Unexpected error: {type(exc).__name__}: {exc}"
            )
            return False
