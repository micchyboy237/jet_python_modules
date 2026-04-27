#!/usr/bin/env python3

"""
This script adds punctuation to text using sherpa-onnx with SenseVoice,
which supports Japanese, Chinese, English, Korean, and Cantonese.

Download model:
https://github.com/k2-fsa/sherpa-onnx/releases/tag/sense-voice

Example:

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/sense-voice/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8.tar.bz2
tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8.tar.bz2
rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8.tar.bz2
"""

import argparse
from pathlib import Path

import sherpa_onnx

MODEL_DIR = Path(
    "/Users/jethroestrada/.cache/pretrained_models/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
)

MODEL_PATH = MODEL_DIR / "model.int8.onnx"
TOKENS_PATH = MODEL_DIR / "tokens.txt"


def create_recognizer() -> sherpa_onnx.OfflineRecognizer:
    if not MODEL_PATH.is_file():
        raise ValueError(f"{MODEL_PATH} does not exist")

    if not TOKENS_PATH.is_file():
        raise ValueError(f"{TOKENS_PATH} does not exist")

    config = sherpa_onnx.OfflineRecognizerConfig(
        model=sherpa_onnx.OfflineModelConfig(
            sense_voice=str(MODEL_PATH),
            tokens=str(TOKENS_PATH),
        ),
        use_itn=True,  # enables punctuation + normalization
    )

    return sherpa_onnx.OfflineRecognizer(config)


def add_punctuation(
    recognizer: sherpa_onnx.OfflineRecognizer,
    text: str,
) -> str:
    """
    Adds punctuation to a given text using SenseVoice.

    Note:
    - Requires sherpa-onnx build that supports `accept_text`
    """

    stream = recognizer.create_stream()

    # Text input (no audio required)
    stream.accept_text(text)

    recognizer.decode_stream(stream)

    result = stream.result.text
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Add punctuation to texts using sherpa-onnx SenseVoice (supports Japanese)."
    )

    parser.add_argument(
        "text_list",
        nargs="*",
        help="List of texts to punctuate. Example: 'text1' 'text2'",
    )

    args = parser.parse_args()

    recognizer = create_recognizer()

    text_list = (
        args.text_list
        if args.text_list
        else [
            "これはテストですよろしくお願いします",
            # "私たちはみんな木のように動かず話すこともできません",
            # "今日はいい天気ですね散歩に行きましょう",
            "我们都是木头人不会说话不会动",
            "The african blogosphere is rapidly expanding bringing more voices online",
        ]
    )

    for text in text_list:
        output = add_punctuation(recognizer, text)

        print("----------")
        print(f"input: {text}")
        print(f"output: {output}")

    print("----------")


if __name__ == "__main__":
    main()
