class QueueObserver:
    """Interface for receiving WebSocket queue status updates."""

    def on_queue_status(
        self, status: str, pending: int, status_color: str, info: dict = None
    ) -> None:
        """Called when queue status changes.

        Args:
            status: Human-readable status text
            pending: Number of segments waiting in queue
            status_color: CSS color for status text
            info: Optional dict with details (segment_num, duration, timestamp, etc.)
        """
        pass

    def on_retry_status(
        self, segment_num: int, attempt: int, delay: float, extra_info: dict = None
    ) -> None:
        """Called when a segment is being retried after failure."""
        pass
