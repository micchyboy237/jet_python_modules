import sounddevice as sd
import numpy as np
import wave
from datetime import datetime
from pathlib import Path

SAMPLE_RATE = 16000
DTYPE = 'int16'


def get_input_channels() -> int:
    device_info = sd.query_devices(sd.default.device[0], 'input')
    return device_info['max_input_channels']


CHANNELS = min(2, get_input_channels())


def record_from_mic(duration: int):
    print(
        f"🎙️ Recording ({CHANNELS} channel{'s' if CHANNELS > 1 else ''}) for {duration} seconds...")
    audio_data = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE
    )
    sd.wait()
    print("✅ Recording complete")
    return audio_data


def save_wav_file(filename, audio_data: np.ndarray):
    filename = Path(filename)  # ✅ ensure Path object
    filename.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(filename), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(np.dtype(DTYPE).itemsize)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
    print(f"💾 Audio saved to {filename}")
