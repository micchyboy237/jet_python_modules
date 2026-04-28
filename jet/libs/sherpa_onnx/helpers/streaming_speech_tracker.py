#!/usr/bin/env python3

"""
This file shows how to remove non-speech segments
and merge all speech segments into a large segment
and save it to a file, using ten-vad.
Segments are saved to disk as soon as they are detected.

Usage

python vad-with-non-streaming-asr.py \
--ten-vad-model /Users/jethroestrada/.cache/pretrained_models/sherpa-onnx/vad/ten-vad.onnx \
--tokens /Users/jethroestrada/.cache/pretrained_models/sherpa-onnx/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17/tokens.txt \
--encoder /Users/jethroestrada/.cache/pretrained_models/sherpa-onnx/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17/encoder-epoch-35-avg-1.int8.onnx \
--decoder /Users/jethroestrada/.cache/pretrained_models/sherpa-onnx/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17/decoder-epoch-35-avg-1.int8.onnx \
--joiner /Users/jethroestrada/.cache/pretrained_models/sherpa-onnx/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17/joiner-epoch-35-avg-1.int8.onnx \
--num-threads 4 \
--sample-rate 16000 \
--feature-dim 80

python vad-with-non-streaming-asr.py \
--ten-vad-model /Users/jethroestrada/.cache/pretrained_models/sherpa-onnx/vad/ten-vad.onnx \
--sense-voice /Users/jethroestrada/.cache/pretrained_models/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx \
--tokens /Users/jethroestrada/.cache/pretrained_models/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
--num-threads 4

Please visit
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx
to download ten-vad.onnx

For instance,

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx
"""

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
import sherpa_onnx
import soundfile as sf

try:
    import colorlog

    _has_colorlog = True
except ImportError:
    _has_colorlog = False

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)

from pathlib import Path

try:
    from tqdm import tqdm

    _has_tqdm = True
except ImportError:
    _has_tqdm = False

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("vad")
    logger.setLevel(logging.DEBUG)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%H:%M:%S"
    if _has_colorlog:
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s" + fmt,
                datefmt=datefmt,
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        )
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(handler)
    return logger


log = setup_logging()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
SEGMENTS_DIR = OUTPUT_DIR / "segments"
SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VAD_MODEL = "/Users/jethroestrada/.cache/pretrained_models/sherpa-onnx/vad/ten-vad.onnx"


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--ten-vad-model",
        type=str,
        default=VAD_MODEL,
        help="Path to ten-vad.onnx",
    )

    return parser.parse_args()


def save_segment(
    seg_num: int,
    samples: np.ndarray,
    start_sample: int,
    sample_rate: int,
) -> Path:
    """Save a speech segment to SEGMENTS_DIR/<seg_num>/sound.wav + segment.json."""
    seg_dir = SEGMENTS_DIR / str(seg_num)
    seg_dir.mkdir(parents=True, exist_ok=True)

    wav_path = seg_dir / "sound.wav"
    sf.write(str(wav_path), samples, samplerate=sample_rate)

    duration_s = len(samples) / sample_rate
    start_s = start_sample / sample_rate
    meta = {
        "segment_number": seg_num,
        "start_sample": start_sample,
        "start_seconds": round(start_s, 4),
        "duration_seconds": round(duration_s, 4),
        "num_samples": len(samples),
        "sample_rate": sample_rate,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "wav_file": "sound.wav",
    }
    json_path = seg_dir / "segment.json"
    json_path.write_text(json.dumps(meta, indent=2))

    log.info(
        "Segment %d saved → %s  |  start=%.2fs  duration=%.2fs  samples=%d",
        seg_num,
        seg_dir,
        start_s,
        duration_s,
        len(samples),
    )
    return seg_dir


def drain_vad_queue(
    vad: sherpa_onnx.VoiceActivityDetector,
    seg_counter: list,  # mutable int box: [count]
    all_speech_samples: list,
    sample_rate: int,
    pbar=None,
) -> None:
    """Drain every ready segment from the VAD queue and save immediately."""
    while not vad.empty():
        segment = vad.front
        samples = np.array(segment.samples, dtype=np.float32)
        start = segment.start
        vad.pop()

        seg_counter[0] += 1
        all_speech_samples.extend(samples)
        save_segment(seg_counter[0], samples, start, sample_rate)

        if pbar is not None:
            pbar.set_postfix(
                segments=seg_counter[0],
                speech_s=f"{len(all_speech_samples) / sample_rate:.1f}s",
            )


def main():
    devices = sd.query_devices()
    if len(devices) == 0:
        log.error("No microphone devices found")
        log.warning(
            "If you are using Linux and you are sure there is a microphone "
            "on your system, please use "
            "./vad-remove-non-speech-segments-alsa.py"
        )
        sys.exit(0)

    default_input_device_idx = sd.default.device[0]
    device_name = devices[default_input_device_idx]["name"]
    log.info("Audio device: %s (index %d)", device_name, default_input_device_idx)

    args = get_args()
    assert_file_exists(args.ten_vad_model)
    log.info("TenVAD model: %s", args.ten_vad_model)

    sample_rate = 16000
    samples_per_read = int(0.1 * sample_rate)

    config = sherpa_onnx.VadModelConfig()
    config.ten_vad.model = args.ten_vad_model
    config.sample_rate = sample_rate

    window_size = config.ten_vad.window_size

    buffer = []
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
    log.info(
        "VAD ready  |  sample_rate=%d  window_size=%d  buffer=30s",
        sample_rate,
        window_size,
    )

    all_samples = []
    all_speech_samples: list = []
    seg_counter = [0]  # mutable int box so drain_vad_queue can mutate it
    total_read_samples = 0

    pbar = None
    if _has_tqdm:
        pbar = tqdm(
            desc="🎙  Listening",
            unit="s",
            bar_format="{desc}: {elapsed}  |  {postfix}",
            dynamic_ncols=True,
        )

    log.info("Started! Please speak. Press Ctrl+C to exit")
    log.info("Segments will be saved to: %s", SEGMENTS_DIR)

    try:
        with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
            while True:
                samples, _ = s.read(samples_per_read)
                samples = samples.reshape(-1)
                total_read_samples += len(samples)

                buffer = np.concatenate([buffer, samples])
                all_samples = np.concatenate([all_samples, samples])

                while len(buffer) > window_size:
                    vad.accept_waveform(buffer[:window_size])
                    buffer = buffer[window_size:]
                    # ← drain the queue immediately after each window
                    drain_vad_queue(
                        vad, seg_counter, all_speech_samples, sample_rate, pbar
                    )

                if pbar is not None:
                    total_s = total_read_samples / sample_rate
                    speech_s = len(all_speech_samples) / sample_rate
                    ratio = speech_s / total_s * 100 if total_s > 0 else 0
                    pbar.set_postfix(
                        total=f"{total_s:.1f}s",
                        speech=f"{speech_s:.1f}s",
                        ratio=f"{ratio:.1f}%",
                        segments=seg_counter[0],
                        refresh=True,
                    )

    except KeyboardInterrupt:
        log.warning("Ctrl+C detected — flushing remaining audio…")
        if pbar is not None:
            pbar.close()

        # Flush any trailing speech the VAD hasn't emitted yet
        vad.flush()
        drain_vad_queue(vad, seg_counter, all_speech_samples, sample_rate, pbar=None)

        # Save merged speech WAV
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if all_speech_samples:
            speech_arr = np.array(all_speech_samples, dtype=np.float32)
            filename_for_speech = OUTPUT_DIR / f"{timestamp}-speech.wav"
            sf.write(str(filename_for_speech), speech_arr, samplerate=sample_rate)
            log.info("Merged speech WAV → %s", filename_for_speech)

        # Save full recording WAV
        filename_for_all = OUTPUT_DIR / f"{timestamp}-all.wav"
        sf.write(str(filename_for_all), all_samples, samplerate=sample_rate)
        log.info("Full recording WAV → %s", filename_for_all)

        total_s = total_read_samples / sample_rate
        speech_s = len(all_speech_samples) / sample_rate
        log.info(
            "Done  |  total=%.1fs  speech=%.1fs  ratio=%.1f%%  segments=%d",
            total_s,
            speech_s,
            speech_s / total_s * 100 if total_s > 0 else 0,
            seg_counter[0],
        )


if __name__ == "__main__":
    main()
