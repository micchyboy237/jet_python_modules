import sounddevice as sd
import numpy as np
import wave
from datetime import datetime
from pathlib import Path

SAMPLE_RATE = 44100
DTYPE = 'int16'

# Adjust this to your desired save location
OUTPUT_DIR = Path(__file__).parent / "generated" / "run_record_mic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Auto-create folder

OUTPUT_FILE = OUTPUT_DIR / \
    f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"


def get_input_channels() -> int:
    """Detect the max input channels of the default device."""
    device_info = sd.query_devices(sd.default.device[0], 'input')
    return device_info['max_input_channels']


CHANNELS = min(2, get_input_channels())  # Use stereo if possible, else mono


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


def save_wav_file(filename: Path, audio_data: np.ndarray):
    filename.parent.mkdir(parents=True, exist_ok=True)  # Ensure dir exists
    with wave.open(str(filename), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(np.dtype(DTYPE).itemsize)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
    print(f"💾 Audio saved to {filename}")


if __name__ == "__main__":
    duration_seconds = 5
    data = record_from_mic(duration_seconds)
    save_wav_file(OUTPUT_FILE, data)
