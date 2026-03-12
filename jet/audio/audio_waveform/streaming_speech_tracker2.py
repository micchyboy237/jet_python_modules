from pathlib import Path

from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig
from jet.audio.audio_waveform.speech_tracker3 import StreamingSpeechTracker

model_dir = str(
    Path("~/.cache/pretrained_models/FireRedVAD/Stream-VAD").expanduser().resolve()
)

vad_config = FireRedStreamVadConfig(
    use_gpu=False,
    smooth_window_size=5,
    speech_threshold=0.5,
    pad_start_frame=5,
    min_speech_frame=30,
    max_speech_frame=1000,  # 10s
    min_silence_frame=20,
    chunk_max_frame=30000,
)
vad = FireRedStreamVad.from_pretrained(model_dir, vad_config)

tracker = StreamingSpeechTracker(vad)

for segment in tracker.run_streaming_audio():
    print(
        "Speech:",
        segment["start_time"],
        "→",
        segment["end_time"],
        "len:",
        len(segment["audio"]),
    )
