#!/usr/bin/env python3

"""
This file shows how to remove non-speech segments
and merge all speech segments into a large segment
and save it to a file.

Usage

python3 ./vad-remove-non-speech-segments-from-file.py \
        --silero-vad-model silero_vad.onnx \
        input.wav \
        output.wav

Please visit
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
to download silero_vad.onnx

For instance,
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import sherpa_onnx
import soundfile as sf

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VAD_MODEL = (
    "/Users/jethroestrada/.cache/pretrained_models/sherpa-onnx/vad/silero_vad.onnx"
)
DEFAULT_AUDIO = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav"


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
        "--silero-vad-model",
        type=str,
        default=VAD_MODEL,
        help=f"Path to silero_vad.onnx. Default: {VAD_MODEL}",
    )

    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=DEFAULT_AUDIO,
        help=f"Path to input.wav. Default: {DEFAULT_AUDIO}",
    )

    parser.add_argument(
        "output",
        type=str,
        nargs="?",
        default=str(OUTPUT_DIR / "output.wav"),
        help="Path to output.wav. If not provided, will be written as 'output.wav' under OUTPUT_DIR.",
    )

    return parser.parse_args()


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def main():
    args = get_args()
    assert_file_exists(args.silero_vad_model)
    assert_file_exists(args.input)

    samples, sample_rate = load_audio(args.input)
    if sample_rate != 16000:
        import librosa

        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = args.silero_vad_model
    config.silero_vad.threshold = 0.5
    config.silero_vad.min_silence_duration = 0.25  # seconds
    config.silero_vad.min_speech_duration = 0.25  # seconds

    # If the current segment is larger than this value, then it increases
    # the threshold to 0.9 internally. After detecting this segment,
    # it resets the threshold to its original value.
    config.silero_vad.max_speech_duration = 5  # seconds

    config.sample_rate = sample_rate

    window_size = config.silero_vad.window_size

    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)

    speech_samples = []
    while len(samples) > window_size:
        vad.accept_waveform(samples[:window_size])
        samples = samples[window_size:]

        while not vad.empty():
            speech_samples.extend(vad.front.samples)
            vad.pop()

    vad.flush()

    while not vad.empty():
        speech_samples.extend(vad.front.samples)
        vad.pop()

    speech_samples = np.array(speech_samples, dtype=np.float32)

    sf.write(args.output, speech_samples, samplerate=sample_rate)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
