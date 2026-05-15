# jet.audio.speech_handlers.api_types

from typing import Literal, Optional, TypedDict


class PhraseSegment(TypedDict):
    """Represents a phrase segment with timing information."""

    phrase: str
    start: float
    end: float


class ClientHeader(TypedDict):
    """
    Header sent as JSON before audio bytes in WebSocket binary message.
    Schema defined in api_schema_live_subtitles_server2.yaml.
    """

    uuid: str
    sample_rate: int
    duration_sec: float
    start_sec: float
    end_sec: float
    vad_reason: Literal["silence", "valley", "hard_limit"]
    forced: bool
    started_at: Optional[str]  # ISO 8601 format


class ServerSuccessResponse(TypedDict):
    """
    Successful response from the live subtitles server.
    Contains transcription, translation, and context information.
    """

    uuid: str
    success: Literal[True]
    transcription_ja: str
    translation_en: str
    context_uuid: str
    context_duration: float
    new_duration: float
    new_ja_similarity: Optional[float]
    new_ja_start_index: Optional[int]
    transcribed_duration_sec: float
    transcribed_duration_pctg: float
    coverage_label: str
    old_ja_sents: list[str]
    new_ja_sents: list[str]
    old_en_sents: list[str]
    new_en_sents: list[str]
    phrase_segments: list[PhraseSegment]


class ServerErrorResponse(TypedDict):
    """
    Error response from the live subtitles server.
    May still contain partial transcription/translation results.
    """

    uuid: str
    error: str
    transcription_ja: str
    translation_en: str


class ServerResponse(TypedDict, total=False):
    """
    Union type for all possible server responses.
    Fields marked total=False to allow both success and error shapes.
    """

    # Common fields
    uuid: str
    transcription_ja: str
    translation_en: str

    # Success-specific fields
    success: bool
    context_uuid: str
    context_duration: float
    new_duration: float
    new_ja_similarity: Optional[float]
    new_ja_start_index: Optional[int]
    transcribed_duration_sec: float
    transcribed_duration_pctg: float
    coverage_label: str
    old_ja_sents: list[str]
    new_ja_sents: list[str]
    old_en_sents: list[str]
    new_en_sents: list[str]
    phrase_segments: list[PhraseSegment]

    # Error-specific fields
    error: str


# Type aliases for convenience
ClientMessageHeader = ClientHeader
SubtitleServerResponse = ServerResponse
