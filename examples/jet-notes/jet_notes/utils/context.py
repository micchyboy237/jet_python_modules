from jet_notes.services.note_service import NoteService
from phoenix.otel import using_metadata, using_session, using_user


def process_request(user_id: str, session_id: str, request_data: dict):
    """Process request with proper context propagation."""

    # Add contextual attributes to all spans in this context
    with (
        using_user(user_id),
        using_session(session_id),
        using_metadata({"request_type": "note_creation"}),
    ):
        service = NoteService()
        # All spans created within this context will have these attributes
        result = service.create_note(
            title=request_data.get("title", "Untitled"),
            content=request_data.get("content", ""),
        )
        return result
