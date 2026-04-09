import asyncio
import ctypes
import os
import queue
import time
from collections import deque

import numpy as np
import sounddevice as sd
import websockets

# ====================== CONFIG ======================
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 320
DTYPE = "int16"
JITTER_BUFFER_SIZE = 8

DEBUG_DECODE = True
DEBUG_QUEUE = True
DEBUG_PLAYBACK = True


def log(level: str, message: str):
    ts = time.strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] [{level.upper()}] {message}")


class OpusDecoder:
    def __init__(self):
        lib_path = "/opt/homebrew/lib/libopus.dylib"
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"libopus.dylib not found at {lib_path}")

        log("INIT", f"Loading Opus decoder from: {lib_path}")
        self.lib = ctypes.CDLL(lib_path)

        self.lib.opus_decoder_create.restype = ctypes.c_void_p
        self.lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
        self.lib.opus_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_short),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.opus_decode.restype = ctypes.c_int

        self.decoder = self.lib.opus_decoder_create(SAMPLE_RATE, CHANNELS)
        if not self.decoder:
            raise RuntimeError("Failed to create Opus decoder")

        log("INIT", "✅ Opus decoder created successfully")

    def decode(self, opus_data: bytes) -> np.ndarray:
        if not opus_data:
            log("DECODE", "Empty opus data")
            return np.zeros((BLOCK_SIZE, CHANNELS), dtype=DTYPE)

        data_ptr = (ctypes.c_ubyte * len(opus_data)).from_buffer_copy(opus_data)
        pcm = (ctypes.c_short * (BLOCK_SIZE * CHANNELS))()

        length = self.lib.opus_decode(
            self.decoder, data_ptr, len(opus_data), pcm, BLOCK_SIZE, 0
        )

        if length < 0:
            log("DECODE", f"ERROR code {length}")
            return np.zeros((BLOCK_SIZE, CHANNELS), dtype=DTYPE)

        # Return as 2D array (frames, channels) - REQUIRED for sounddevice
        arr = np.ctypeslib.as_array(pcm, shape=(BLOCK_SIZE, CHANNELS)).astype(DTYPE)

        if DEBUG_DECODE:
            rms = np.sqrt(np.mean(arr.astype(float) ** 2))
            log(
                "DECODE",
                f"SUCCESS → {len(opus_data)} bytes → {BLOCK_SIZE} samples | RMS={rms:.2f}",
            )

        return arr


class ReceiverPipeline:
    def __init__(self):
        self.decoder = OpusDecoder()
        self.jitter_buffer = deque(maxlen=JITTER_BUFFER_SIZE)
        self.last_seq = -1
        self.output_queue = queue.Queue(maxsize=100)
        self.last_queue_log = time.time()
        log("INIT", "✅ ReceiverPipeline initialized")

    def decode_packet(self, raw_msg: bytes):
        try:
            parts = raw_msg.split(b"|", 1)
            if len(parts) != 2:
                log("PACKET", f"Invalid format ({len(parts)} parts)")
                return

            seq = int(parts[0])
            opus_data = parts[1]

            log("PACKET", f"Received seq={seq} | opus_size={len(opus_data)} bytes")

            if seq > self.last_seq:
                pcm = self.decoder.decode(opus_data)
                self.output_queue.put_nowait(pcm)
                self.last_seq = seq

                if DEBUG_QUEUE and time.time() - self.last_queue_log > 2.0:
                    log("QUEUE", f"size={self.output_queue.qsize()}")
                    self.last_queue_log = time.time()
        except Exception as e:
            log("ERROR", f"decode_packet failed: {e}")


pipeline = ReceiverPipeline()


async def websocket_receiver(websocket):
    log("CONN", f"Client connected from {websocket.remote_address}")
    try:
        async for message in websocket:
            pipeline.decode_packet(message)
    except Exception as e:
        log("CONN", f"Connection closed: {e}")


def playback_callback(outdata, frames, time_info, status):
    if status:
        log("PLAYBACK", f"Status: {status}")

    try:
        chunk = pipeline.output_queue.get_nowait()
        # Ensure shape is (frames, channels) = (320, 1)
        if chunk.ndim == 1:
            chunk = chunk.reshape(-1, 1)
        outdata[:] = chunk

        if DEBUG_PLAYBACK:
            rms = np.sqrt(np.mean(chunk.astype(float) ** 2))
            log("PLAYBACK", f"Played {frames} samples | RMS={rms:.2f}")
    except queue.Empty:
        outdata[:] = 0
        if DEBUG_PLAYBACK:
            log("PLAYBACK", "Silence (queue empty)")


async def main():
    log("START", "Starting WebSocket server on port 8765")
    async with websockets.serve(websocket_receiver, "0.0.0.0", 8765):
        log("START", "✅ Server is running")
        log("INFO", "→ Route BlackHole output correctly")

        try:
            with sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=BLOCK_SIZE,
                latency="low",
                callback=playback_callback,
            ):
                log("AUDIO", "OutputStream started")
                await asyncio.Future()
        except Exception as e:
            log("ERROR", f"Audio error: {e}")


if __name__ == "__main__":
    log("START", "Receiver starting on Mac Mini")
    asyncio.run(main())
