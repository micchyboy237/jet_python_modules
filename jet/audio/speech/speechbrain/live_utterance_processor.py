import json
import os
import shutil
import uuid
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from jet.audio.speech.speechbrain.utterance_processor import StreamingSpeechProcessor
from jet.audio.speech.speechbrain.ws_client import PersistentWSClient
from rich.console import Console

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
SEGMENTS_OUTPUT_DIR = OUTPUT_DIR / "segments"
SEGMENTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
console = Console()
SAMPLE_RATE = 16000
CHUNK_SIZE = 16000
WS_URI = os.getenv("LOCAL_WS_LIVE_SUBTITLES_URL", "ws://localhost:8765")
CLIENT_ID = "mac_client_1"
processor = StreamingSpeechProcessor(
    sampling_rate=SAMPLE_RATE,
    threshold=0.5,
    neg_threshold=0.25,
    max_speech_duration_sec=8.0,
)
ws_client = PersistentWSClient(WS_URI)
ws_client.start()
segment_counter = 0


def _save_segment(segment_dict: dict, audio_buffer: np.ndarray):
    """
    Save:
        segments/segment_0001/
            ├── sound.wav
            └── segment.json
    """
    global segment_counter
    segment_counter += 1
    segment_id = f"segment_{segment_counter:04d}"
    segment_dir = SEGMENTS_OUTPUT_DIR / segment_id
    segment_dir.mkdir(parents=True, exist_ok=True)
    start_sec = float(segment_dict["start"])
    end_sec = float(segment_dict["end"])
    start_sample = int(start_sec * SAMPLE_RATE)
    end_sample = int(end_sec * SAMPLE_RATE)
    segment_audio = audio_buffer[start_sample:end_sample]
    wav_path = segment_dir / "sound.wav"
    sf.write(wav_path, segment_audio, SAMPLE_RATE)
    metadata = {
        "segment_id": segment_id,
        "type": segment_dict.get("type"),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": float(segment_dict.get("duration", end_sec - start_sec)),
        "probability": float(segment_dict.get("prob", 0.0)),
        "sampling_rate": SAMPLE_RATE,
        "num_samples": int(len(segment_audio)),
    }
    json_path = segment_dir / "segment.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    console.print(
        f"[bold green]Saved:[/bold green] {segment_id} "
        f"({start_sec:.2f}s - {end_sec:.2f}s)"
    )
    segment_int16 = np.clip(segment_audio, -1.0, 1.0)
    segment_int16 = (segment_int16 * 32767.0).astype(np.int16)
    return segment_int16.tobytes()


def audio_callback(indata, frames, time, status):
    if status:
        console.print(status)
    mono = np.mean(indata, axis=1).astype(np.float32)
    previous_buffer = processor.utterance_audio_buffer.copy()
    payload = processor.process(mono)
    if payload["submitted_count"] > 0:
        console.print("[bold green]Speech completed![/bold green]")
        console.print(payload)
        full_audio = np.concatenate([previous_buffer, mono])
        utterance_id = str(uuid.uuid4())
        for segment_counter, seg in enumerate(payload["speech_segments"]):
            audio_bytes = _save_segment(seg, full_audio)
            ws_client.send_audio(audio_bytes, CLIENT_ID, utterance_id, segment_counter)


with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    blocksize=CHUNK_SIZE,
    dtype="float32",
    callback=audio_callback,
):
    console.print("[bold cyan]Listening... Press Ctrl+C to stop[/bold cyan]")
    sd.sleep(60_000)
