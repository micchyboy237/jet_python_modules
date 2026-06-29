"""
Color/rating helpers, CSS constants, and entry-formatting for SubtitleOverlay.
Extracted from subtitle_overlay_window.py to keep the window class under 150 lines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

OVERLAY_WIDTH = 450
OVERLAY_HEIGHT = 600

_VAD_BADGE_HTML = (
    '<span style="'
    "background:#3b1f6b; color:#c084fc; font-family:monospace; "
    "font-size:9px; font-weight:bold; padding:1px 5px; "
    'border-radius:3px; letter-spacing:0.5px;">FRD</span>'
)


def _balanced_vad_score_color(score: Optional[float]) -> str:
    if score is None:
        return "#8b949e"
    if score >= 0.85:
        return "#3fb950"
    if score >= 0.70:
        return "#56d364"
    if score >= 0.55:
        return "#e3b341"
    if score >= 0.40:
        return "#fb923c"
    return "#f85149"


def _balanced_vad_score_rating(score: Optional[float]) -> str:
    if score is None:
        return "N/A"
    if score >= 0.85:
        return "Excellent"
    if score >= 0.70:
        return "Good"
    if score >= 0.55:
        return "Marginal"
    if score >= 0.40:
        return "Poor"
    return "Invalid"


def _speech_pctg_color(pctg: Optional[float]) -> str:
    if pctg is None:
        return "#8b949e"
    if pctg < 30:
        return "#f85149"
    if pctg < 50:
        return "#fb923c"
    if pctg < 70:
        return "#e3b341"
    return "#3fb950"


def _composite_score_color(score: Optional[float]) -> str:
    if score is None:
        return "#8b949e"
    if score > 0.80:
        return "#3fb950"
    if score > 0.60:
        return "#56d364"
    if score > 0.40:
        return "#e3b341"
    if score > 0.20:
        return "#fb923c"
    return "#f85149"


def _quality_label_color(label: Optional[str]) -> str:
    _MAP = {
        "Very good": "#3fb950",
        "Good": "#56d364",
        "Fair": "#e3b341",
        "Bad": "#fb923c",
        "Very bad": "#f85149",
    }
    return _MAP.get(label or "", "#8b949e")


def _speaker_confidence_color(confidence: Optional[float]) -> str:
    if confidence is None:
        return "#8b949e"
    if confidence > 0.70:
        return "#3fb950"
    if confidence > 0.50:
        return "#e3b341"
    if confidence > 0.30:
        return "#fb923c"
    return "#f85149"


def _format_entry(
    index: int,
    entry: dict,
    prev_entry: Optional[dict],
    hide_japanese: bool,
    expanded: bool = False,
    is_playing: bool = False,
) -> str:
    segment_number = entry["segment_number"]
    ja = entry.get("ja", "").strip()
    en = entry.get("en", "").strip()
    text_display = en if hide_japanese else f"{ja}\n{en}".strip()
    start: float = entry.get("start", 0.0)
    end: float = entry.get("end", 0.0)
    end_reason = entry.get("end_reason") or "true_silence"
    segment_dir: Optional[Path] = (
        Path(entry["segment_dir"]) if entry.get("segment_dir") else None
    )
    gap: Optional[float] = None
    if prev_entry is not None:
        prev_end_time_utc = prev_entry.get("end_time_utc")
        prev_end_sec = prev_entry.get("end")
        current_start_time_utc = entry.get("start_time_utc")
        current_start_sec = entry.get("start")
        if current_start_time_utc and prev_end_time_utc:
            from datetime import datetime  # local import kept to avoid circular deps

            try:
                start_dt = datetime.fromisoformat(current_start_time_utc)
                end_dt = datetime.fromisoformat(prev_end_time_utc)
                gap = (start_dt - end_dt).total_seconds()
            except (ValueError, TypeError):
                pass
        if gap is None and current_start_sec is not None and prev_end_sec is not None:
            gap = current_start_sec - prev_end_sec
    gap_str = f"{gap:.2f}s" if gap is not None else "—"
    duration = end - start
    vad_score: Optional[float] = entry.get("vad_score")
    vad_score_str = f"{vad_score:.3f}" if isinstance(vad_score, float) else "N/A"
    vad_score_color = _balanced_vad_score_color(vad_score)
    vad_rating = _balanced_vad_score_rating(vad_score)
    speech_pctg: Optional[float] = entry.get("speech_frames_pctg")
    speech_pctg_str = (
        f"{speech_pctg:.1f}%" if isinstance(speech_pctg, (int, float)) else "N/A"
    )
    speech_pctg_color = _speech_pctg_color(speech_pctg)
    transcribed_pctg = entry.get("transcribed_duration_pctg")
    trans_pctg_str = (
        f"{float(transcribed_pctg):.1f}%"
        if isinstance(transcribed_pctg, (int, float))
        else "N/A"
    )
    trans_pctg_color = _speech_pctg_color(
        float(transcribed_pctg) if isinstance(transcribed_pctg, (int, float)) else None
    )
    composite_score: Optional[float] = entry.get("vad_composite_score")
    composite_str = (
        f"{composite_score:.3f}" if isinstance(composite_score, (int, float)) else "N/A"
    )
    composite_color = _composite_score_color(
        float(composite_score) if isinstance(composite_score, (int, float)) else None
    )
    quality_label: Optional[str] = entry.get("vad_quality_label")
    quality_str = quality_label or "N/A"
    quality_color = _quality_label_color(quality_label)
    speaker_label = entry.get("speaker_label", "")
    speaker_confidence: Optional[float] = entry.get("speaker_confidence")
    speaker_conf_str = (
        f" ({speaker_confidence:.2f})" if isinstance(speaker_confidence, float) else ""
    )
    speaker_conf_color = _speaker_confidence_color(speaker_confidence)
    speaker_match_type = entry.get("speaker_match_type", "")
    copy_link = (
        f'<a href="copy:{index}" style="color:#58a6ff; text-decoration:none;">📋</a>'
    )
    open_link = (
        f'<a href="open:{index}" style="color:#58a6ff; text-decoration:none;">📂</a>'
        if segment_dir
        else ""
    )
    play_icon = "⏸" if is_playing else "▶"
    play_link = (
        f'<a href="play:{index}" style="color:#ff7b72; text-decoration:none;'
        f' font-size:13px;">{play_icon}</a>'
        if segment_dir and (segment_dir / "sound.wav").exists()
        else ""
    )
    speaker_badge = ""
    if speaker_label:
        speaker_badge = (
            f' <span style="'
            f"background:#1f3b5c; color:#79c0ff; font-family:monospace; "
            f"font-size:9px; font-weight:bold; padding:1px 5px; "
            f'border-radius:3px;">{speaker_label}'
            f'<span style="color:{speaker_conf_color};">{speaker_conf_str}</span>'
            f"</span>"
        )
    header_html = (
        f'<b style="font-size:10px;">{segment_number}</b>'
        f"{speaker_badge} "
        f'<span style="font-size:9px; color:#8b949e;">'
        f"({duration:.2f}s)"
        f" • gap: {gap_str}"
        f' • <span style="color:#d2a8ff;">{end_reason}</span>'
        f' • VAD: <span style="color:{vad_score_color}; font-weight:bold;">{vad_score_str}</span>'
        f' <span style="color:{vad_score_color}; font-size:8px;">({vad_rating})</span>'
        f"</span>"
        f" {copy_link} {open_link} {play_link}"
    )
    text_html = (
        f'<div style="margin-top:5px; margin-bottom:2px;">'
        f'<span style="'
        f"font-size:13px; "
        f"color:#e6edf3; "
        f"font-family:'Segoe UI', 'SF Pro Text', Arial, sans-serif; "
        f"line-height:1.5; "
        f"letter-spacing:0.5px;"
        f'">{text_display.replace(chr(10), "<br/>")}</span>'
        f"</div>"
    )
    return (
        f'<div style="margin-bottom:4px;">'
        f"{header_html}"
        f"{text_html}"
        f"</div>"
        f'<hr style="border:none; border-top:1px solid #30363d; margin:4px 0;">'
    )
