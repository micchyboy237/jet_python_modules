# confidence.py
from typing import TypedDict, List, Dict

class DiarizationSegment(TypedDict):
    start: float
    end: float
    speaker: str
    confidence: Dict[str, float]

class LowConfidenceSegment(TypedDict):
    start: float
    end: float
    speaker: str
    confidence: float

class VoiceprintConfidence(TypedDict):
    speaker: str
    confidence: Dict[str, float]

class HighConfidenceMatch(TypedDict):
    speaker: str
    match: str
    confidence: float

def collect_low_confidence_segments(
    diarization: Dict[str, List[DiarizationSegment]],
    thresh: float = 70.0
) -> List[LowConfidenceSegment]:
    low_segments: List[LowConfidenceSegment] = []
    for seg in diarization["segments"]:
        seg_conf = seg["confidence"][seg["speaker"]]
        if seg_conf < thresh:
            low_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg["speaker"],
                "confidence": seg_conf,
            })
    return sorted(low_segments, key=lambda x: x["confidence"])

def filter_high_confidence(
    ident_result: Dict[str, List[VoiceprintConfidence]],
    threshold: float = 60.0
) -> List[HighConfidenceMatch]:
    high: List[HighConfidenceMatch] = []
    for vp in ident_result["voiceprints"]:
        label, score = max(vp["confidence"].items(), key=lambda x: x[1])
        if score >= threshold:
            high.append({
                "speaker": vp["speaker"],
                "match": label,
                "confidence": score,
            })
    return high