import time

import numpy as np
import sounddevice as sd
import torch
from jet.audio.audio_waveform.vad.firered import FireRedVADWrapper

# ────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────

SAMPLE_RATE = 16000
BLOCK_SIZE_MS = 30  # process audio in ~30 ms blocks
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_SIZE_MS / 1000)

CHANNELS = 1
DEVICE = None  # None = default input device
# DEVICE = "USB Audio Device"   # or explicit device name/index

# How often to check for speech segments (seconds)
CHECK_INTERVAL = 0.15

# ────────────────────────────────────────────────


def callback(indata: np.ndarray, frames: int, time_info, status):
    """Called by sounddevice for every new audio block"""
    if status:
        print("Audio callback warning:", status, file=sys.stderr)

    # indata shape: (frames, channels)
    mono = indata[:, 0] if indata.ndim > 1 else indata
    vad.audio_buffer = np.concatenate([vad.audio_buffer, mono.astype(np.float32)])


# ────────────────────────────────────────────────
#  Main streaming logic
# ────────────────────────────────────────────────

print("Initializing FireRedVAD wrapper...")
vad = FireRedVADWrapper(device="cuda" if torch.cuda.is_available() else "cpu")

print(f"Starting audio input @ {SAMPLE_RATE} Hz, block = {BLOCK_SIZE_MS} ms")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    channels=CHANNELS,
    # dtype="float32",
    # latency="low",
    # device=DEVICE,
    callback=callback,
):
    print("Recording...  Press Ctrl+C to stop")

    last_check = time.time()

    try:
        while True:
            now = time.time()

            # Check for completed speech segments periodically
            if now - last_check >= CHECK_INTERVAL:
                result = vad.get_speech_segments()

                if result is not None:
                    frame_results, (start_sec, end_sec), current_time = result

                    print(
                        f"[{current_time:6.2f}s] "
                        f"Speech segment detected: {start_sec:6.2f} – {end_sec:6.2f}s "
                        f"({end_sec - start_sec:4.2f}s long)"
                    )

                    # Optional: you can also inspect frame_results[-5:] etc.
                    # or send the segment (vad.audio_buffer slice) to ASR here

                last_check = now

            # Very light sleep — we don't want to block the callback
            time.sleep(0.020)

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        print("Stream closed.")
        # Optional: process any remaining buffered audio
        final_result = vad.get_speech_segments()
        if final_result:
            _, (s, e), t = final_result
            print(f"Final trailing segment: {s:.2f} – {e:.2f}s")
