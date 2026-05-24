# jet.audio.speech_handlers.api_types

from typing import Literal, Optional, TypedDict

from jet.audio.audio_waveform.vad._types import SpeechSegment


class WordSegment(TypedDict):
    """Individual word with timing information."""

    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    word: str


class PhraseSegment(TypedDict):
    """Represents a phrase segment with timing information and word-level detail."""

    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    phrase: str
    word_segments: list[WordSegment]


class SpeakerInfo(TypedDict):
    """Information about a known speaker."""

    label: str
    segment_count: int
    first_seen: float
    last_seen: float
    active_duration: float
    has_valid_centroid: bool


class Diarization(TypedDict):
    """Speaker diarization information."""

    current_speaker: str
    known_speakers: list[str]
    speaker_count: int
    speakers_info: dict[str, SpeakerInfo]
    total_segments_processed: int


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
    # NEW: Absolute UTC timestamps (ISO 8601 strings)
    start_time_utc: Optional[str]  # e.g. "2026-05-25T14:32:17.123456+00:00"
    end_time_utc: Optional[str]  # e.g. "2026-05-25T14:32:19.876543+00:00"
    # Real-time gap from previous segment end to this segment start
    gap_sec: Optional[float]  # e.g. 2.35 (seconds), None for first segment


class _BaseResponseFields(TypedDict, total=False):
    """Shared fields between success and error responses."""

    uuid: str
    transcription_ja: str
    translation_en: str


class ServerSuccessResponse(_BaseResponseFields):
    """
    Successful response from the live subtitles server.
    Contains transcription, translation, and context information.
    """

    success: Literal[True]
    context_uuid: str
    context_duration: float
    new_duration: float
    new_ja_similarity: Optional[float]
    new_ja_start_index: Optional[int]
    transcribed_duration_sec: float
    transcribed_duration_pctg: float
    coverage_label: str
    speaker_label: str
    speaker_confidence: float
    speaker_match_type: str
    diarization: Diarization
    old_ja_sents: list[str]
    new_ja_sents: list[str]
    old_en_sents: list[str]
    new_en_sents: list[str]
    phrase_segments: list[PhraseSegment]


class ServerErrorResponse(_BaseResponseFields):
    """
    Error response from the live subtitles server.
    May still contain partial transcription/translation results.
    """

    error: str


class ServerResponse(_BaseResponseFields, total=False):
    """
    Union type for all possible server responses.
    Fields marked total=False to allow both success and error shapes.
    """

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
    speaker_label: str
    speaker_confidence: float
    speaker_match_type: str
    diarization: Diarization
    old_ja_sents: list[str]
    new_ja_sents: list[str]
    old_en_sents: list[str]
    new_en_sents: list[str]
    phrase_segments: list[PhraseSegment]

    # Error-specific fields
    error: str


class _SubtitleResponseFields(_BaseResponseFields):
    """All possible response fields for subtitle notifications."""

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
    speaker_label: str
    speaker_confidence: float
    speaker_match_type: str
    diarization: Diarization
    old_ja_sents: list[str]
    new_ja_sents: list[str]
    old_en_sents: list[str]
    new_en_sents: list[str]
    phrase_segments: list[PhraseSegment]

    # Error-specific fields
    error: str


class SubtitleNotification(_SubtitleResponseFields, total=False):
    """
    Payload passed to subtitle observers. Combines the server response
    with locally‑available segment metadata so the UI can display everything.
    """

    # --- local segment metadata ---
    segment: SpeechSegment
    num: int
    start_sec: float
    end_sec: float
    end_reason: str
    segment_dir: str  # path as string; UI converts to Path
    avg_vad_prob: Optional[float]
    speech_frames_pctg: Optional[float]
    speech_dur_sec: Optional[float]
    # NEW: Pass through absolute timestamps
    start_time_utc: Optional[str]
    end_time_utc: Optional[str]


# Type aliases for convenience
ClientMessageHeader = ClientHeader
SubtitleServerResponse = ServerResponse
