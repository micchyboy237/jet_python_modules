import time
from dataclasses import dataclass

from opentelemetry import trace

tracer = trace.get_tracer(__name__)


@dataclass
class Note:
    id: int
    title: str
    content: str


class NoteService:
    def __init__(self):
        self._next_id = 1

    def create_note(self, title: str, content: str) -> Note:
        # Create a custom span for this operation
        with tracer.start_as_current_span("create_note") as span:
            span.set_attribute("note.title", title)
            span.set_attribute("note.content_length", len(content))

            try:
                note = self._save_to_database(title, content)
                span.set_attribute("note.id", note.id)
                span.set_status(trace.StatusCode.OK)
                return note
            except Exception as e:
                # Record the error in the span
                span.record_exception(e)
                span.set_status(trace.StatusCode.ERROR, str(e))
                raise

    def _save_to_database(self, title: str, content: str) -> Note:
        """Simulate database latency and save."""
        with tracer.start_as_current_span("db_save_note") as db_span:
            db_span.set_attribute("db.operation", "insert")
            db_span.set_attribute("db.system", "mock_sqlite")

            # Simulate network/db latency
            time.sleep(0.1)

            note = Note(id=self._next_id, title=title, content=content)
            self._next_id += 1

            db_span.set_attribute("db.row_count", 1)
            return note
