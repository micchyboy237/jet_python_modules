import asyncio
import ctypes
import queue
from collections import deque

import numpy as np
import sounddevice as sd
import websockets

# ============== CONFIG ==============
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 512
DTYPE = "int16"
JITTER_BUFFER_SIZE = 8


# ============== DIRECT OPUS DECODER ==============
class OpusDecoder:
    def __init__(self):
        self.lib = ctypes.CDLL("/opt/homebrew/lib/libopus.dylib")
        self.decoder = self.lib.opus_decoder_create(SAMPLE_RATE, CHANNELS)

    def decode(self, opus_data: bytes) -> np.ndarray:
        pcm = (ctypes.c_short * (BLOCK_SIZE * CHANNELS))()
        length = self.lib.opus_decode(
            self.decoder, opus_data, len(opus_data), pcm, BLOCK_SIZE, 0
        )
        if length < 0:
            return np.zeros((BLOCK_SIZE, CHANNELS), dtype=DTYPE)
        return np.ctypeslib.as_array(pcm, shape=(BLOCK_SIZE, CHANNELS)).astype(DTYPE)

    def __del__(self):
        if hasattr(self, "decoder"):
            self.lib.opus_decoder_destroy(self.decoder)


# ============== PIPELINE ==============
class ReceiverPipeline:
    def __init__(self):
        self.decoder = OpusDecoder()
        self.jitter_buffer = deque(maxlen=JITTER_BUFFER_SIZE)
        self.last_seq = -1
        self.output_queue = queue.Queue(maxsize=50)

    def decode_packet(self, raw_msg: bytes):
        try:
            parts = raw_msg.split(b"|", 2)
            if len(parts) != 3:
                return
            seq = int(parts[0])
            opus_data = parts[2]

            if seq > self.last_seq:
                pcm = self.decoder.decode(opus_data)
                self.output_queue.put_nowait(pcm)
                self.last_seq = seq
        except Exception as e:
            print("Decode error:", e)


# ============== RECEIVER ==============
pipeline = ReceiverPipeline()


async def websocket_receiver(websocket):
    print("✅ Receiver ready – Listening with TEN-VAD sourced audio")
    async for message in websocket:
        pipeline.decode_packet(message)


def playback_callback(outdata, frames, time_info, status):
    if status:
        print("Output status:", status)
    try:
        chunk = pipeline.output_queue.get_nowait()
        outdata[:] = chunk
    except queue.Empty:
        outdata.fill(0)


async def main():
    async with websockets.serve(websocket_receiver, "0.0.0.0", 8765):
        print("WebSocket server running on port 8765")
        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE,
            latency="low",
            callback=playback_callback,
        ):
            await asyncio.Future()


if __name__ == "__main__":
    print("Starting receiver on Mac Mini.")
    print("→ Route Python output to BlackHole Aggregate Device in Audio MIDI Setup.")
    print("→ Select the Aggregate as system Input in apps like Zoom.")
    asyncio.run(main())
