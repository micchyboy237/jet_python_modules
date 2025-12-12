from pathlib import Path

from jet.audio.speech.pyannote.utils import export_plotly_timeline

# Example: Minimal realistic data from a 2-speaker meeting
sample_turns = [
    {
        "segment_index": 0,
        "speaker": "SPEAKER_00",
        "start_sec": 0.0,
        "end_sec": 12.45,
        "duration_sec": 12.45,
        "confidence": 0.9421,
        "wav_path": "segments/segment_0000/segment.wav"
    },
    {
        "segment_index": 1,
        "speaker": "SPEAKER_01",
        "start_sec": 12.8,
        "end_sec": 28.1,
        "duration_sec": 15.3,
        "confidence": 0.8976,
        "wav_path": "segments/segment_0001/segment.wav"
    },
    {
        "segment_index": 2,
        "speaker": "SPEAKER_00",
        "start_sec": 29.0,
        "end_sec": 45.7,
        "duration_sec": 16.7,
        "confidence": 0.9654,
        "wav_path": "segments/segment_0002/segment.wav"
    },
    # ... more turns
]

if __name__ == "__main__":
    output_dir = Path("./example_output")
    output_dir.mkdir(exist_ok=True)

    export_plotly_timeline(
        turns=sample_turns,
        total_seconds=187.3,                    # total audio duration in seconds
        audio_name="team_meeting_20251212.wav", # appears in title
        output_dir=output_dir,
    )

    print(f"Open this file in your browser: {output_dir / 'timeline.html'}")