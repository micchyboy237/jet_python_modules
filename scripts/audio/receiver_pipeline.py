import asyncio
import ctypes
import os
import queue
import time

import numpy as np
import sounddevice as sd
import websockets

# ====================== CONFIG ======================
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 320  # Must match sender's OPUS_FRAME_SIZE
DTYPE = "int16"
DEBUG_DECODE = True
DEBUG_PLAYBACK = True
DEBUG_QUEUE = False  # Set True if you want queue spam


# ====================== STRUCTURED LOGGING ======================
def log(level: str, message: str):
    """Clean timestamped logging for easy traceability"""
    ts = time.strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] [{level.upper():<8}] {message}")


class OpusDecoder:
    def __init__(self):
        lib_path = "/opt/homebrew/lib/libopus.dylib"
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"libopus.dylib not found at {lib_path}")

        log("INIT", f"Loading Opus decoder from: {lib_path}")
        self.lib = ctypes.CDLL(lib_path)

        # Set function signatures safely for macOS arm64
        self.lib.opus_decoder_create.restype = ctypes.c_void_p
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
        """Decode Opus → PCM with proper ctypes pointer"""
        if not opus_data:
            log("DECODE", "Received empty opus data")
            return np.zeros((BLOCK_SIZE, CHANNELS), dtype=DTYPE)

        # Critical: bytes → ctypes pointer
        data_ptr = (ctypes.c_ubyte * len(opus_data)).from_buffer_copy(opus_data)
        pcm_buffer = (ctypes.c_short * (BLOCK_SIZE * CHANNELS))()

        length = self.lib.opus_decode(
            self.decoder, data_ptr, len(opus_data), pcm_buffer, BLOCK_SIZE, 0
        )

        if length < 0:
            log(
                "DECODE",
                f"ERROR: opus_decode returned {length} (input={len(opus_data)} bytes)",
            )
            return np.zeros((BLOCK_SIZE, CHANNELS), dtype=DTYPE)

        # Return as (frames, channels) shape for sounddevice
        pcm = np.ctypeslib.as_array(pcm_buffer, shape=(BLOCK_SIZE, CHANNELS)).astype(
            DTYPE
        )

        if DEBUG_DECODE:
            rms = np.sqrt(np.mean(pcm.astype(float) ** 2))
            log(
                "DECODE",
                f"SUCCESS → {len(opus_data)} bytes → {BLOCK_SIZE} samples | RMS={rms:.2f}",
            )

        return pcm


class ReceiverPipeline:
    def __init__(self):
        self.decoder = OpusDecoder()
        self.last_seq = -1
        self.output_queue = queue.Queue(maxsize=120)  # generous buffer
        self.last_queue_log = time.time()
        log("INIT", "✅ ReceiverPipeline initialized")

    def decode_packet(self, raw_msg: bytes):
        """Parse and decode incoming WebSocket packet"""
        try:
            if not raw_msg:
                log("PACKET", "Empty message received")
                return

            parts = raw_msg.split(b"|", 1)
            if len(parts) != 2:
                log("PACKET", f"Invalid format: {len(parts)} parts (expected seq|data)")
                return

            seq = int(parts[0])
            opus_data = parts[1]

            log("PACKET", f"Received seq={seq} | size={len(opus_data)} bytes")

            if seq > self.last_seq:
                pcm = self.decoder.decode(opus_data)
                self.output_queue.put_nowait(pcm)
                self.last_seq = seq

                if DEBUG_QUEUE and time.time() - self.last_queue_log > 2.0:
                    log("QUEUE", f"size={self.output_queue.qsize()}")
                    self.last_queue_log = time.time()
            else:
                log("PACKET", f"Ignored duplicate/out-of-order seq={seq}")

        except Exception as e:
            log("ERROR", f"decode_packet failed: {e}")


# Global pipeline instance
pipeline = ReceiverPipeline()


async def websocket_receiver(websocket):
    """WebSocket handler with clean lifecycle logging"""
    peer = websocket.remote_address
    log("CONN", f"Client connected from {peer}")

    try:
        async for message in websocket:
            pipeline.decode_packet(message)
    except websockets.exceptions.ConnectionClosedOK:
        log("CONN", "Connection closed cleanly by sender")
    except websockets.exceptions.ConnectionClosedError as e:
        log("CONN", f"Connection closed with error: {e}")
    except Exception as e:
        log("ERROR", f"WebSocket handler error: {e}")
    finally:
        log("CONN", "WebSocket receiver handler ended")


def playback_callback(outdata, frames, time_info, status):
    """Sounddevice output callback"""
    if status:
        log("PLAYBACK", f"Status: {status}")

    try:
        chunk = pipeline.output_queue.get_nowait()
        # Ensure correct shape (frames, channels)
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
    log("START", "Starting TEN-VAD Receiver on port 8765")

    async with websockets.serve(websocket_receiver, "0.0.0.0", 8765):
        log("START", "✅ WebSocket server is running")
        log("INFO", "→ Make sure BlackHole Aggregate Device is set as output")
        log("INFO", "→ Select BlackHole as input in Zoom/Teams/etc.")

        try:
            with sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=BLOCK_SIZE,
                latency="low",
                callback=playback_callback,
            ):
                log("AUDIO", "OutputStream started successfully")
                await asyncio.Future()  # keep running
        except Exception as e:
            log("ERROR", f"Audio output error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
