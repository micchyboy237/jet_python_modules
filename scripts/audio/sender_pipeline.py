import asyncio
import ctypes
import queue
import time

import numpy as np
import sounddevice as sd
import websockets
from ten_vad import VAD  # Modern TEN-VAD import

# ============== CONFIG ==============
HOST = "192.168.x.x"  # ← CHANGE TO YOUR MAC MINI'S LOCAL IP
PORT = 8765
SAMPLE_RATE = 16000  # TEN-VAD works best at 16kHz
CHANNELS = 1
BLOCK_SIZE = 512  # ~32ms at 16kHz (adjust for latency; TEN-VAD handles well)
DTYPE = "int16"
VAD_THRESHOLD = 0.5  # Probability threshold (tune based on testing)
OPUS_BITRATE = 24000  # Lower for voice at 16kHz


# ============== DIRECT OPUS ENCODER (ctypes) ==============
class OpusEncoder:
    def __init__(self):
        self.lib = ctypes.CDLL("/opt/homebrew/lib/libopus.dylib")
        self.encoder = self.lib.opus_encoder_create(
            SAMPLE_RATE, CHANNELS, 2049, None
        )  # VOIP mode
        self.lib.opus_encoder_ctl(self.encoder, 4002, ctypes.c_int(OPUS_BITRATE))

    def encode(self, pcm: np.ndarray) -> bytes:
        pcm_bytes = pcm.astype(np.int16).tobytes()
        max_data_bytes = len(pcm_bytes) * 2
        data = (ctypes.c_ubyte * max_data_bytes)()
        length = self.lib.opus_encode(
            self.encoder,
            pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),
            len(pcm),
            data,
            max_data_bytes,
        )
        return bytes(data[:length])

    def __del__(self):
        if hasattr(self, "encoder"):
            self.lib.opus_encoder_destroy(self.encoder)


# ============== PIPELINE ==============
class AudioPipeline:
    def __init__(self):
        self.vad = VAD()  # TEN-VAD instance (lightweight, real-time)
        self.encoder = OpusEncoder()
        self.audio_queue = queue.Queue(maxsize=50)
        self.sequence = 0

    def pre_process(self, indata: np.ndarray) -> np.ndarray:
        # Resample/downmix if needed; here we assume 16kHz mono from sounddevice
        return indata.astype(np.float32) / 32768.0

    def is_speech(self, frame: np.ndarray) -> bool:
        # TEN-VAD frame-level detection (expects raw int16 or float; adjust if needed)
        # Many bindings accept numpy array directly and return prob or bool
        prob = (
            self.vad.detect(frame)
            if hasattr(self.vad, "detect")
            else self.vad.process(frame)[0]
        )
        return prob >= VAD_THRESHOLD

    def encode(self, pcm: np.ndarray) -> bytes:
        return self.encoder.encode(pcm)


# ============== SENDER ==============
pipeline = AudioPipeline()


def audio_callback(indata, frames, time_info, status):
    if status:
        print("Input status:", status)

    processed = pipeline.pre_process(
        indata.copy().flatten()
    )  # Flatten for frame processing

    if pipeline.is_speech(indata.copy()):  # Use original int16 for VAD/encoding
        encoded = pipeline.encode(indata.copy())
        packet = {"seq": pipeline.sequence, "ts": time.time(), "data": encoded}
        pipeline.audio_queue.put_nowait(packet)
        pipeline.sequence += 1


async def websocket_sender():
    async with websockets.connect(f"ws://{HOST}:{PORT}") as ws:
        print("✅ Sender connected – Streaming with TEN-VAD + direct Opus")
        while True:
            try:
                packet = pipeline.audio_queue.get_nowait()
                msg = (
                    str(packet["seq"]).encode()
                    + b"|"
                    + str(packet["ts"]).encode()
                    + b"|"
                    + packet["data"]
                )
                await ws.send(msg)
            except queue.Empty:
                await asyncio.sleep(0.001)


def run_sender():
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=BLOCK_SIZE,
        latency="low",
        callback=audio_callback,
    ):
        asyncio.run(websocket_sender())


if __name__ == "__main__":
    print("Starting sender on MacBook Air with TEN-VAD.")
    print(
        "Tip: Create BlackHole aggregate (Built-in Mic + BlackHole) for clean routing."
    )
    run_sender()
