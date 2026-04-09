import asyncio
import ctypes
import os
import platform
import time

import numpy as np
import sounddevice as sd
import websockets
from ten_vad import TenVad

# ====================== CONFIG ======================
HOST = "192.168.68.30"
PORT = 8765

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

INPUT_BLOCK_SIZE = 1024
VAD_HOP_SIZE = 256
VAD_THRESHOLD = 0.5

HANGOVER_FRAMES = 12
OPUS_BITRATE = 24000
OPUS_FRAME_SIZE = 320

LIBOPUS_PATH = "/opt/homebrew/lib/libopus.dylib"

DEBUG_VAD = True
DEBUG_OPUS = True
DEBUG_SEND = True


# ====================== LOGGING ======================
def log(level: str, message: str):
    ts = time.strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] [{level}] {message}")


class OpusEncoder:
    def __init__(self):
        if not os.path.exists(LIBOPUS_PATH):
            raise FileNotFoundError(f"libopus.dylib not found at {LIBOPUS_PATH}")

        log("INIT", f"Loading Opus from: {LIBOPUS_PATH}")
        self.lib = ctypes.CDLL(LIBOPUS_PATH)

        self.lib.opus_encoder_create.restype = ctypes.c_void_p
        self.lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
        self.lib.opus_encoder_ctl.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.opus_encode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_short),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int32,
        ]
        self.lib.opus_encode.restype = ctypes.c_int32

        error = ctypes.c_int()
        self.encoder = self.lib.opus_encoder_create(
            SAMPLE_RATE, CHANNELS, 2049, ctypes.byref(error)
        )
        if not self.encoder or error.value != 0:
            raise RuntimeError(f"Opus encoder failed: {error.value}")

        self.lib.opus_encoder_ctl(self.encoder, 4002, OPUS_BITRATE)
        log("INIT", f"✅ Opus encoder ready (frame={OPUS_FRAME_SIZE} samples)")

    def encode(self, pcm: np.ndarray) -> bytes:
        if pcm.ndim == 2:
            pcm = pcm.flatten()
        pcm = np.ascontiguousarray(pcm.astype(np.int16)).flatten()

        if DEBUG_OPUS:
            log(
                "OPUS",
                f"Input {len(pcm)} samples | min={pcm.min():.0f} max={pcm.max():.0f} mean={pcm.mean():.1f}",
            )

        encoded = bytearray()
        for i in range(0, len(pcm), OPUS_FRAME_SIZE):
            chunk = pcm[i : i + OPUS_FRAME_SIZE]
            if len(chunk) < OPUS_FRAME_SIZE:
                if DEBUG_OPUS:
                    log("OPUS", f"Padding last chunk {len(chunk)} → {OPUS_FRAME_SIZE}")
                chunk = np.pad(
                    chunk, (0, OPUS_FRAME_SIZE - len(chunk)), mode="constant"
                )

            max_data = OPUS_FRAME_SIZE * 2 + 256
            data = (ctypes.c_ubyte * max_data)()

            length = self.lib.opus_encode(
                self.encoder,
                chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),
                OPUS_FRAME_SIZE,
                data,
                ctypes.c_int32(max_data),
            )

            if length > 0:
                encoded.extend(data[:length])
                if DEBUG_OPUS:
                    log("OPUS", f"Chunk OK → {length} bytes")
            else:
                log("OPUS", f"ERROR code {length}")
                return b""

        if DEBUG_OPUS and encoded:
            log("OPUS", f"✓ Total encoded: {len(encoded)} bytes")
        return bytes(encoded)


class SenderPipeline:
    def __init__(self):
        log("INIT", "Starting SenderPipeline...")
        try:
            self.vad = TenVad(hop_size=VAD_HOP_SIZE, threshold=VAD_THRESHOLD)
            log("INIT", "✅ TEN VAD loaded")
        except Exception as e:
            log("ERROR", f"TEN VAD failed: {e}")
            raise

        self.encoder = OpusEncoder()
        self.ws = None
        self.seq = 0
        self.is_speaking = False
        self.hangover_counter = 0
        self.loop = None
        log("INIT", "✅ SenderPipeline ready")

    async def connect_websocket(self):
        uri = f"ws://{HOST}:{PORT}"
        log("CONN", f"Connecting to {uri}")
        self.ws = await websockets.connect(uri)
        self.loop = asyncio.get_running_loop()
        log("CONN", "✅ Connected to receiver")

    def process_vad_frame(self, frame: np.ndarray) -> bool:
        if len(frame) != VAD_HOP_SIZE:
            return False
        prob, flag = self.vad.process(frame)
        if DEBUG_VAD:
            log(
                "VAD",
                f"prob={prob:.4f} flag={flag} speaking={self.is_speaking} hangover={self.hangover_counter}",
            )
        return flag == 1

    def should_send_audio(self) -> bool:
        if self.is_speaking:
            self.hangover_counter = HANGOVER_FRAMES
            return True
        elif self.hangover_counter > 0:
            self.hangover_counter -= 1
            return True
        return False

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            log("AUDIO", f"Input status: {status}")

        audio_block = indata.flatten().astype(DTYPE)

        sent_this_block = False

        for i in range(0, len(audio_block), VAD_HOP_SIZE):
            hop = audio_block[i : i + VAD_HOP_SIZE]
            if len(hop) < VAD_HOP_SIZE:
                break

            is_voice = self.process_vad_frame(hop)

            if is_voice:
                self.is_speaking = True

            if self.should_send_audio() and not sent_this_block:
                opus_data = self.encoder.encode(audio_block)
                if opus_data and self.ws:
                    # Safe send using the event loop from main thread
                    if self.loop and not self.loop.is_closed():
                        self.loop.call_soon_threadsafe(
                            lambda data=opus_data: asyncio.create_task(
                                self.send_packet(data)
                            )
                        )
                        sent_this_block = True
                        self.seq += 1
                    else:
                        log("SEND", "WARNING: No running loop - cannot send")

        if not is_voice and self.hangover_counter == 0:
            self.is_speaking = False

    async def send_packet(self, opus_data: bytes):
        if not self.ws:
            return
        try:
            packet = f"{self.seq}|".encode() + opus_data
            await self.ws.send(packet)
            if DEBUG_SEND:
                log("SEND", f"Sent seq={self.seq} | {len(opus_data)} bytes")
        except Exception as e:
            log("SEND", f"Failed: {e}")

    async def run(self):
        await self.connect_websocket()
        log("START", "Microphone + TEN VAD started. Speak now!")

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=INPUT_BLOCK_SIZE,
                latency="low",
                callback=self.audio_callback,
            ):
                await asyncio.Future()
        except Exception as e:
            log("ERROR", f"Stream error: {e}")
        finally:
            if self.ws:
                try:
                    await self.ws.close()
                except:
                    pass
            log("SHUTDOWN", "Sender stopped")


if __name__ == "__main__":
    log("START", "TEN-VAD Sender Pipeline Starting")
    log("INFO", f"Target receiver: {HOST}:{PORT}")
    log("INFO", f"Python arch: {platform.machine()}")

    pipeline = SenderPipeline()
    try:
        asyncio.run(pipeline.run())
    except KeyboardInterrupt:
        log("SHUTDOWN", "Stopped by user")
    except Exception as e:
        log("ERROR", f"Unexpected error: {e}")
