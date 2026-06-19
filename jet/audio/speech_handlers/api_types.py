from typing import Literal, Optional, TypedDict

from jet.audio.audio_waveform.vad._types import SpeechSegment

# ---------------------------------------------------------------------------
# Shared / atomic types
# ---------------------------------------------------------------------------


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


class SpeakerResult(TypedDict):
    """Individual speaker identification result (from API schema)."""

    label: str
    confidence: float
    match_type: str
    is_primary: bool
    is_new_speaker: bool


# ---------------------------------------------------------------------------
# Client → Server  (WebSocket binary message header)
# ---------------------------------------------------------------------------

LanguageCode = Literal["auto", "en", "ja", "zh", "ko"]
VadReason = Literal["silence", "valley", "hard_limit"]


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
    vad_reason: VadReason
    forced: bool
    started_at: Optional[str]
    start_time_utc: Optional[str]
    end_time_utc: Optional[str]
    gap_sec: Optional[float]
    vad_score: Optional[float]
    language: LanguageCode
    # ── new fields ──
    segment_number: int
    segment_id: str


# ---------------------------------------------------------------------------
# Server → Client  (WebSocket text response)
# ---------------------------------------------------------------------------


class _BaseResponseFields(TypedDict, total=False):
    """Shared fields between success and error responses."""

    uuid: str
    ja_text: str
    en_text: str


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
    # ── new fields from API schema ──
    language: str
    event: str
    emo: str
    speaker_labeling_performed: bool
    speakers: list[SpeakerResult]
    segment_number: int
    segment_dir: str


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
    error: str
    # ── new fields ──
    language: str
    event: str
    emo: str
    speaker_labeling_performed: bool
    speakers: list[SpeakerResult]
    segment_number: int
    segment_dir: str


# ---------------------------------------------------------------------------
# SubtitleNotification  (enriched response sent to observers)
# ---------------------------------------------------------------------------


class _SubtitleResponseFields(_BaseResponseFields):
    """All possible response fields for subtitle notifications."""

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
    error: str
    # ── new fields ──
    language: str
    event: str
    emo: str
    speaker_labeling_performed: bool
    speakers: list[SpeakerResult]
    segment_number: int
    segment_dir: str


class SubtitleNotification(_SubtitleResponseFields, total=False):
    """
    Payload passed to subtitle observers. Combines the server response
    with locally‑available segment metadata so the UI can display everything.
    """

    segment: SpeechSegment
    num: int
    start_sec: float
    end_sec: float
    end_reason: str
    avg_vad_prob: Optional[float]
    vad_score: Optional[float]
    speech_frames_pctg: Optional[float]
    speech_dur_sec: Optional[float]
    start_time_utc: Optional[str]
    end_time_utc: Optional[str]


# Backward-compatible aliases
ClientMessageHeader = ClientHeader
SubtitleServerResponse = ServerResponse
