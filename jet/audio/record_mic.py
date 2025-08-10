import sounddevice as sd
import numpy as np
import wave
from datetime import datetime

# Configurable parameters
SAMPLE_RATE = 44100   # CD quality
CHANNELS = 2          # Stereo
DTYPE = 'int16'       # Standard WAV format
OUTPUT_FILE = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"


def record_from_mic(duration: int):
    """
    Records audio from the default microphone for the given duration (in seconds).
    Returns the audio data as a numpy array.
    """
    print(f"🎙️ Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                        channels=CHANNELS, dtype=DTYPE)
    sd.wait()  # Wait until recording finishes
    print("✅ Recording complete")
    return audio_data


def save_wav_file(filename: str, audio_data: np.ndarray):
    """
    Saves the recorded numpy audio data to a WAV file.
    """
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(np.dtype(DTYPE).itemsize)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
    print(f"💾 Audio saved to {filename}")


if __name__ == "__main__":
    duration_seconds = 10  # Change to how long you want to record
    data = record_from_mic(duration_seconds)
    save_wav_file(OUTPUT_FILE, data)
