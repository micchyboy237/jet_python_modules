from html import escape
from typing import List

from jet.audio.speech.speechbrain.speech_types import SpeechSegment


def build_speech_segments_table(segments: List[SpeechSegment]) -> str:
    """
    Build an HTML table for speech segments.
    Generic, reusable, presentation-only.
    """

    header = """
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Start (ms)</th>
                <th>End (ms)</th>
                <th>Duration (ms)</th>
                <th>Prob</th>
                <th>Frame Start</th>
                <th>Frame End</th>
                <th>Type</th>
            </tr>
        </thead>
        <tbody>
    """

    rows: list[str] = []

    for seg in segments:
        duration = seg["end"] - seg["start"]

        rows.append(
            f"""
            <tr>
                <td>{seg["num"]}</td>
                <td>{seg["start"]:.1f}</td>
                <td>{seg["end"]:.1f}</td>
                <td>{duration:.1f}</td>
                <td>{seg["prob"]:.3f}</td>
                <td>{seg["frame_start"]}</td>
                <td>{seg["frame_end"]}</td>
                <td class="type-{escape(seg["type"])}">
                    {escape(seg["type"])}
                </td>
            </tr>
            """
        )

    footer = "</tbody></table>"

    return header + "".join(rows) + footer
